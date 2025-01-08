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

            # Get embeddings for FAISS search
            from utils.FAISS_utils import query_with_context, load_faiss_index, load_metadata
            index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            metadata = load_metadata(CONFIG.METADATA_PATH)

            # Use CLIP to search for similar images
            results = query_with_context(
                index=index,
                metadata=metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                image_query=image,
                top_k=CONFIG.DEFAULT_TOP_K
            )

            # Extract document references and contexts
            document_refs = set()
            contexts = []
            similar_images = []

            if results and results[0]:
                for result_group in results:
                    for result in result_group:
                        metadata = result["metadata"]
                        similarity = 1 - (result['distance'] / 2)

                        if metadata.get('type') == 'image' and similarity > self.similarity_threshold:
                            # Get image data and metadata
                            image_id = metadata.get('image', {}).get('id')
                            if image_id:
                                image_data = self._prepare_image_data(image_id, metadata, similarity)
                                if image_data:
                                    similar_images.append(image_data)
                                    source_doc = metadata.get('image', {}).get('source_doc', '')
                                    if source_doc:
                                        document_refs.add(source_doc)

                        elif metadata.get('type') == 'text-chunk':
                            if 'get_content' in metadata:
                                chunk_text = metadata['get_content']()
                                if chunk_text:
                                    contexts.append(chunk_text)
                            source_path = metadata.get('path', '')
                            if source_path:
                                document_refs.add(source_path)

            # Deduplicate similar images
            if similar_images:
                similar_images = self.image_classifier.deduplicate(
                    similar_images,
                    self.deduplication_threshold
                )

            # Analyze technical aspects with context
            technical_context = await self.analyze_technical_context({
                'image': image,
                'base64_image': base64_image
            })

            # Build query context including FAISS results
            query_context = self.prompt_loader.format_template(
                'image_query_with_context',
                query_text=query_text or "Analyze this technical image",
                image_context="\n".join([
                    technical_context.get('analysis', ''),
                    *contexts
                ])
            )

            # Process with GPT, including document references
            vision_result = await self._process_vision_request(
                base64_image,
                query_context,
                list(document_refs)  # Pass document references
            )

            # Format the response
            formatted_response = self.formatter.format_response(vision_result['response'])

            return {
                'response': formatted_response,
                'technical_context': technical_context,
                'confidence': classification_result['confidence'],
                'related_images': similar_images,
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

    async def _process_vision_request(self, base64_image: str, query_context: str, document_refs: List[str]) -> Dict:
        """Process vision request with GPT with retries and proper formatting."""
        MAX_RETRIES = 3
        BASE_WAIT = 4

        for attempt in range(MAX_RETRIES):
            try:
                # Process document references to extract IDs
                processed_refs = []
                for doc_ref in document_refs:
                    # Extract filename from path
                    filename = os.path.basename(doc_ref)
                    # Extract ID (characters before first dash)
                    match = re.match(r'^([A-Za-z0-9]+)-', filename)
                    if match:
                        doc_id = match.group(1)
                        processed_refs.append(doc_id)

                # Format document references
                doc_refs_text = "\n".join(
                    [f"[ref]{doc_id}[/ref]" for doc_id in processed_refs]
                )
                doc_context = (
                    f"\nRelevant Documentation (use document IDs in [ref] tags):\n{doc_refs_text}"
                    if processed_refs else ""
                )

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

                                    {doc_context}

                                    IMPORTANT: When referencing documents:
                                    - Use ONLY the document ID (characters before first dash)
                                    - Format as [ref]DOCUMENT_ID[/ref]
                                    - Example: from "PE12A000E-RZ104-AIO rel 010.pdf" use [ref]PE12A000E[/ref]
                                    - Do not include paths or full filenames
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
