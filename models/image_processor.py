# models/image_processor.py
import base64
import logging
import random
import asyncio
from io import BytesIO
from typing import Dict, Optional, List
from PIL import Image
from openai import OpenAI
from fastapi import HTTPException

from config import CONFIG
from utils.img_utils import ImageStore, ImageClassifier
from models.prompt_manager import PromptLoader


class ImageProcessor:
    """Handles image processing and analysis for the RAG system."""

    def __init__(self, openai_client: OpenAI, model, processor, device, formatter):
        """Initialize the image processor with necessary components."""
        self.client = openai_client
        self.model = model
        self.processor = processor
        self.device = device
        self.image_store = ImageStore()
        self.image_classifier = ImageClassifier(
            model=model,
            processor=processor,
            device=device
        )
        self.prompt_loader = PromptLoader()
        self.formatter = formatter  # Use the formatter passed from RAGQueryServer
        self.labels = ["a technical image", "a non-technical image"]
        self.similarity_threshold = CONFIG.IMAGE_SIMILARITY_THRESHOLD
        self.technical_threshold = CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD
        self.deduplication_threshold = CONFIG.DEDUPLICATION_THRESHOLD

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> Dict:
        """Process an image query and generate comprehensive analysis."""
        try:
            # Preprocess and classify image
            image = self._preprocess_image(image_data)
            base64_image = self._convert_to_base64(image)

            classification_result = await self._classify_image(image)
            if not classification_result['is_technical']:
                # Format non-technical response
                formatted_response = self.formatter.format_response(
                    classification_result['response']
                )
                return {
                    **classification_result,
                    'response': formatted_response
                }

            # Analyze technical aspects
            technical_context = await self.analyze_technical_context({
                'image': image,
                'base64_image': base64_image
            })

            # Build query context
            query_context = self.prompt_loader.format_template(
                'image_query_with_context',
                query_text=query_text or "Analyze this technical image",
                image_context=technical_context.get('analysis', '')
            )

            # Process with GPT
            vision_result = await self._process_vision_request(base64_image, query_context)

            # Format the response using the formatter from RAGQueryServer
            formatted_response = self.formatter.format_response(vision_result['response'])

            return {
                'response': formatted_response,
                'technical_context': technical_context,
                'confidence': classification_result['confidence'],
                'related_images': vision_result.get('related_images', [])
            }

        except Exception as e:
            logging.error(f"Error in process_image_query: {e}")
            raise

    async def _classify_image(self, image: Image.Image) -> Dict:
        """Classify image as technical or non-technical."""
        predicted_label, confidence = self.image_classifier.classify(
            image=image,
            labels=self.labels
        )

        return {
            'is_technical': predicted_label == "a technical image" and confidence > self.technical_threshold,
            'confidence': confidence,
            'response': "This image does not appear to be a technical diagram or component."
            if predicted_label != "a technical image" or confidence <= self.technical_threshold
            else None
        }

    async def analyze_technical_context(self, image_data: Dict) -> Dict:
        """Extract and analyze technical context from image."""
        try:
            base64_image = image_data.get('base64_image') or self._convert_to_base64(image_data['image'])

            messages = [
                {
                    "role": "assistant",
                    "content": self.prompt_loader.get_system_prompt('vision_assistant')
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this technical image and identify key components and specifications."
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=messages,
                max_tokens=150
            )

            return {
                "system_category": "UV Water Treatment System",
                "components_list": "",
                "documentation_refs": "",
                "maintenance_notes": "",
                "analysis": response.choices[0].message.content
            }

        except Exception as e:
            logging.error(f"Error in analyze_technical_context: {e}")
            return {
                "system_category": "Unknown",
                "components_list": "",
                "documentation_refs": "",
                "maintenance_notes": "",
                "analysis": ""
            }

    def get_relevant_images(self, results: List[Dict]) -> List[Dict]:
        """Get and process relevant images from search results."""
        relevant_images = []
        try:
            for result in results:
                metadata = result['metadata']
                similarity = 1 - (result['distance'] / 2)

                if metadata.get('type') == 'image' and similarity > self.similarity_threshold:
                    image_id = (metadata.get('image', {}).get('id') or
                                metadata.get('content', {}).get('image_id'))

                    if not image_id:
                        continue

                    image_data = self._prepare_image_data(image_id, metadata, similarity)
                    if image_data:
                        relevant_images.append(image_data)

            return self.image_classifier.deduplicate(relevant_images, self.deduplication_threshold)

        except Exception as e:
            logging.error(f"Error getting relevant images: {e}")
            return []

    async def _process_vision_request(self, base64_image: str, query_context: str) -> Dict:
        """Process vision request with GPT with retries and proper formatting."""
        MAX_RETRIES = 3
        BASE_WAIT = 4

        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {
                        "role": "assistant",
                        "content": self.prompt_loader.get_system_prompt('vision_assistant')
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                    {query_context}

                                    IMPORTANT: If this component appears in our documentation, 
                                    explicitly mention that and reference the document using double quotes.
                                    Example: This component appears in ""Document_Name"" where it is described as...

                                    Use the provided documentation context to enhance your analysis.
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ]

                response = self.client.chat.completions.create(
                    model=CONFIG.GPT_VISION_MODEL,
                    messages=messages,
                    max_tokens=CONFIG.VISION_MAX_TOKENS
                )

                if not response or not response.choices:
                    raise ValueError("Empty or invalid response from OpenAI API")

                # Get the raw response
                raw_response = response.choices[0].message.content.strip()

                return {
                    "response": raw_response,
                    "related_images": []
                }

            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_WAIT * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"API error, retrying in {wait_time:.2f}s ({attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait_time)
                    continue
                raise HTTPException(
                    status_code=500,
                    detail=f"Vision request failed after {MAX_RETRIES} attempts: {str(e)}"
                )

    def _preprocess_image(self, image_data: bytes) -> Image.Image:
        """Preprocess image data into PIL Image."""
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise

    def _convert_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _prepare_image_data(self, image_id: str, metadata: Dict, similarity: float) -> Optional[Dict]:
        """Prepare image data from metadata."""
        try:
            base64_image = self.image_store.get_base64(image_id)
            if not base64_image:
                return None

            return {
                'image': base64_image,
                'image_id': image_id,
                'caption': metadata.get('image', {}).get('caption', ''),
                'context': metadata.get('context', ''),
                'source': str(metadata.get('path', '')),
                'similarity': similarity
            }
        except Exception as e:
            logging.error(f"Error preparing image data: {e}")
            return None
