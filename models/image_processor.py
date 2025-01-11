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
import os
import re

from config import CONFIG
from utils.img_utils import ImageStore, ImageClassifier
from models.prompt_manager import PromptLoader


class ImageProcessor:
    """Handles image processing and analysis for the RAG system."""

    def __init__(self, openai_client: OpenAI, model, device, formatter):
        """Initialize the image processor with necessary components."""
        self.client = openai_client
        self.model = model
        self.device = device
        self.image_store = ImageStore()
        self.image_classifier = ImageClassifier(
            model=model,
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
                formatted_response = self.formatter.format_response(
                    classification_result['response']
                )
                return {
                    **classification_result,
                    'response': formatted_response,
                    'similar_images': []  # Add empty list for consistency
                }

            # Get embeddings for FAISS search
            from utils.FAISS_utils import query_with_context, load_faiss_index, load_metadata
            index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            metadata = load_metadata(CONFIG.METADATA_PATH)

            # Use CLIP to search for similar images
            results = query_with_context(
                index=index,
                metadata=metadata,
                model=self.model,
                device=self.device,
                image_query=image,
                top_k=CONFIG.DEFAULT_TOP_K
            )

            # Extract document references and process similar images
            document_refs = set()
            contexts = []  # This was previously unused
            similar_images = []

            if results and results[0]:
                for result_group in results:
                    for result in result_group:
                        metadata = result["metadata"]
                        similarity = 1 - (result['distance'] / 2)

                        if metadata.get('type') == 'image' and similarity > self.similarity_threshold:
                            image_id = metadata.get('image', {}).get('id')
                            if image_id:
                                image_data = self._prepare_image_data(image_id, metadata, similarity)
                                if image_data:
                                    similar_images.append(image_data)
                                    source_doc = metadata.get('image', {}).get('source_doc', '')
                                    if source_doc:
                                        filename = os.path.basename(source_doc)
                                        match = re.match(r'^([A-Za-z0-9]+)-', filename)
                                        if match:
                                            document_refs.add(match.group(1))

                        elif metadata.get('type') == 'text-chunk':
                            if 'get_content' in metadata:
                                chunk_text = metadata['get_content']()
                                if chunk_text:
                                    contexts.append(chunk_text)  # Collecting text contexts

            # Deduplicate similar images
            if similar_images:
                similar_images = self.image_classifier.deduplicate(
                    similar_images,
                    self.deduplication_threshold
                )
                logging.info(f"Found {len(similar_images)} similar images after deduplication")

            # Build query context including both technical analysis and text contexts
            query_context = self.prompt_loader.format_template(
                'image_query_with_context',
                query_text=query_text or "Analyze this technical image",
                image_context="\n\n".join([
                    "Technical Documentation Context:",
                    *contexts  # Include the collected text contexts
                ]) if contexts else "No additional context available."
            )

            # Process with GPT
            vision_result = await self._process_vision_request(
                base64_image,
                query_context,  # Now includes both technical analysis and text contexts
                list(document_refs)
            )

            # Format the response
            formatted_response = self.formatter.format_response(vision_result['response'])

            return {
                'response': formatted_response,
                'is_technical': True,
                'confidence': classification_result['confidence'],
                'similar_images': [
                    {
                        'image': img['image'],
                        'caption': img['caption'],
                        'context': img['context'],
                        'source': img['source'],
                        'similarity': str(img['similarity'])  # Convert to string for JSON
                    } for img in similar_images
                ],
                'document_references': list(document_refs)
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
        """
        Extract and analyze technical context from image using GPT Vision model.

        Args:
            image_data: Dictionary containing either:
                - 'base64_image': Base64 encoded image string
                - 'image': PIL Image object

        Returns:
            Dict containing:
                - system_category: Type of system (e.g., "UV Water Treatment System")
                - components_list: List of identified components
                - documentation_refs: Related documentation references
                - maintenance_notes: Any maintenance considerations
                - analysis: Detailed analysis from GPT Vision
        """
        try:
            # Handle image input and convert to base64 if needed
            if 'base64_image' in image_data:
                base64_image = image_data['base64_image']
            else:
                image = image_data.get('image')
                if not isinstance(image, Image.Image):
                    raise ValueError("Invalid image format")
                base64_image = self._convert_to_base64(image)

            # Prepare the message for GPT Vision API
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

            # Call GPT Vision API
            response = self.client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=messages,
                max_tokens=150
            )

            # Return structured analysis
            return {
                "system_category": "UV Water Treatment System",
                "components_list": "",
                "documentation_refs": "",
                "maintenance_notes": "",
                "analysis": response.choices[0].message.content
            }

        except Exception as e:
            logging.error(f"Error in analyze_technical_context: {e}")
            logging.error("Full traceback:", exc_info=True)
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

    async def _process_vision_request(
            self,
            base64_image: str,
            query_context: str,
            document_refs: List[str]
    ) -> Dict:
        """Process vision request with GPT with retries and proper formatting."""
        MAX_RETRIES = 3
        BASE_WAIT = 4

        for attempt in range(MAX_RETRIES):
            try:
                # Format document references list if any exist
                doc_context = ""
                if document_refs:
                    doc_refs_text = "\n".join([f"- [ref]{doc_id}[/ref]" for doc_id in document_refs])
                    doc_context = f"\nRelevant Documentation:\n{doc_refs_text}"

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
                                    Analyze this technical image with the following context:
                                    {query_context}

                                    {doc_context}

                                    IMPORTANT: When referencing documents:
                                    - Use ONLY the document ID (characters before first dash)
                                    - Format as [ref]DOCUMENT_ID[/ref]
                                    - Do not include paths or full filenames

                                    Provide a comprehensive analysis including:
                                    1. Component identification
                                    2. Technical specifications visible in the image
                                    3. Integration with UV system
                                    4. Related documentation references
                                    5. Any safety or maintenance considerations
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

    @staticmethod
    def _preprocess_image(image_data: bytes) -> Image.Image:
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

    @staticmethod
    def _convert_to_base64(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        Handles RGBA and other image modes by converting to RGB with white background.
        """
        try:
            # Handle RGBA images
            if image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                # Paste using alpha channel as mask
                background.paste(image, mask=image.split()[3])
                image = background
            # Handle other non-RGB modes
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error converting image to base64: {e}")
            raise

    def _prepare_image_data(self, image_id: str, metadata: Dict, similarity: float) -> Optional[Dict]:
        """Prepare image data from metadata."""
        try:
            base64_image = self.image_store.get_base64(image_id)
            if not base64_image:
                return None

            return {
                'image': base64_image,  # Base64 encoded image data
                'image_id': image_id,
                'caption': metadata.get('image', {}).get('caption', '') or 'Similar image',
                'context': metadata.get('context', '') or 'Related technical documentation',
                'source': str(metadata.get('path', '')),
                'similarity': float(similarity)  # Ensure similarity is a float
            }
        except Exception as e:
            logging.error(f"Error preparing image data: {e}")
            return None
