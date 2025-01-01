"""
This module implements a FastAPI server for a Retrieval-Augmented Generation (RAG) system.

The server uses a combination of FAISS, PyTorch, OpenAI GPT models, and custom utilities
to process text and image queries. It provides endpoints for document upload, query processing,
and chat-based retrieval functionalities. Other features include proper logging, CORS configuration,
document management, and metadata handling for image-text interactions.

Main Components:
- FastAPI setup and routing for handling API requests.
- Integration with FAISS for efficient vector-based similarity searches.
- Enhanced prompt building with GPT transformers for generating responses.
- Metadata gathering and zero-shot classification for technical images.
- Image deduplication and storage management.
- Custom logging for system operations and document management.
- Configuration management for easy setup and customization of the RAG system.

Note: This server is designed to be run as a standalone application, not as a module.
It should be executed directly and not imported as a library.

Classes and methods:
- Server: The main class that sets up and runs the FastAPI application.
- RAGSystem: Manages the core functionality of the RAG system, including document processing,
  query handling, and response generation.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import field
from datetime import datetime
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from urllib.parse import unquote
import random

import faiss
from PIL import Image
from dotenv import load_dotenv
from fastapi import Body
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from config import CONFIG
from models.prompt_manager import PromptLoader, PromptBuilder
from utils.FAISS_utils import load_faiss_index, load_metadata, query_with_context
from utils.LLM_utils import CLIP_init, openai_post_request
from utils.document_utils import (
    remove_document_from_rag,
    delete_folder_from_rag,
    rename_folder_in_rag,
    create_folder,
    validate_folder_name,
    rescan_documents,
)
from utils.img_utils import ImageStore, ImageClassifier
from models.agents.agent_manager import AgentManager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            CONFIG.LOG_PATH / "system.log",
            maxBytes=CONFIG.MAX_LOG_SIZE,
            backupCount=CONFIG.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
    ]
)


def clean_path(path: str) -> str:
    """
    Sanitizes a file path by removing traversal sequences, decoding URL encoding,
    and replacing separators with the system-specific separator.

    Args:
        path (str): The path to be cleaned.

    Returns:
        str: The sanitized file path.
    """
    # First, decode any URL encoding
    decoded = unquote(path)
    # Remove any potential path traversal
    cleaned = decoded.replace('..', '').strip('/').strip('\\')
    # Normalize path separators
    normalized = cleaned.replace('\\', os.path.sep).replace('/', os.path.sep)
    return normalized


class NoCacheStaticFiles(StaticFiles):
    """
    A subclass of FastAPI's StaticFiles to disable caching for static file responses.
    Ensures that the client always gets the latest version of static files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        response = await super().__call__(scope, receive, send)
        if hasattr(response, 'headers'):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response


# Data models
class ChatMessage(BaseModel):
    """
    Represents a single chat message with role and content.

    Attributes:
        role (str): The role of the sender, such as 'user' or 'assistant'.
        content (str): The content of the message.
    """
    role: str
    content: str


class ChatHistory(BaseModel):
    messages: List[ChatMessage]


class QueryType(BaseModel):
    is_overview: bool = False
    is_technical: bool = False
    is_summary: bool = False


class QueryResponse(BaseModel):
    text_response: str
    images: List[Dict[str, Any]] = field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self):
        return {
            "text_response": self.text_response,
            "images": [{
                "image": img.get("image", ""),
                "caption": img.get("caption", ""),
                "context": img.get("context", ""),
                "source": img.get("source", ""),
                "similarity": str(img.get("similarity", 0))  # Convert float to string
            } for img in self.images]
        }


class EnhancedResponseFormatter:

    def __init__(self):
        self.prompt_builder = PromptBuilder()

    def prepare_prompt(self, query_text: str, contexts: List[str], query_type: QueryType, images: List[Dict]) -> str:
        return self.prompt_builder.build_chat_prompt(query_text, contexts, images, [], query_type.is_technical)

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        return self.prompt_builder.build_messages(prompt)

    @staticmethod
    def format_response(content: str) -> str:

        def process_document_references(text: str) -> str:
            try:
                # Load processed files
                with open("processed_files.json", 'r', encoding='utf-8') as f:
                    processed_files = json.load(f)

                def find_matching_path(doc_ref: str) -> str:
                    # Remove spaces from the reference
                    search_term = doc_ref.replace(' ', '')
                    # logging.info(f"Looking for document reference: {search_term}")

                    for file_path in processed_files:
                        # Normalize path separators
                        norm_path = file_path.replace('\\', '/')
                        clean_path = norm_path.replace(' ', '')
                        if search_term in clean_path:
                            # Extract path relative to Raw Documents
                            if 'Raw Documents/' in norm_path:
                                relative_path = norm_path.split('Raw Documents/')[1]
                                # logging.info(f"Found matching path: {relative_path}")
                                return relative_path
                    # logging.info(f"No matching path found for {search_term}")
                    return ''

                # Find and replace document references
                pattern = r'""([^"]+)""'

                def replacement(match):
                    doc_ref = match.group(1)
                    rel_path = find_matching_path(doc_ref)
                    if rel_path:
                        # Create an onclick handler that calls openDocument
                        return f'<a href="javascript:void(0)" onclick="openDocument(\'{rel_path}\')" class="doc-link">{doc_ref}</a>'
                    # Return just the reference text without double-double quotes if no match found
                    return doc_ref

                # Replace all document references
                text = re.sub(pattern, replacement, text)
                return text

            except Exception as e:
                logging.error(f"Error processing document references: {e}")
                return text

        def clean_text(text: str) -> str:
            # Clean up excess whitespace while preserving structure
            text = re.sub(r'\s*\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()

        def format_lists(content: str) -> str:
            # Add <br> before valid numbered list items
            content = re.sub(r'([^\n])\s*(\d+\.\s+(?=[A-Za-z]))', r'\1<br>\2', content)
            # Add <br> before headers
            content = re.sub(r'([^\n])\s*(#{2,3}\s+)', r'\1<br>\2', content)
            # Format bullet points with proper indentation
            content = re.sub(r'(?m)^[•\-]\s*', r'  • ', content)
            # Remove bullet points from bold text items with bullets
            content = re.sub(r'\*\*\s*•\s*', r'• ', content)
            # Format numbered lists with proper indentation
            content = re.sub(r'(?m)^(\d+\.\s+)(\*\*.*?\*\*)', r'    \1\2', content)
            # Ensure line breaks between list items
            content = re.sub(r'(?<!<br>)(\d+\.\s+)(\*\*.*?\*\*)', r'<br>\1\2', content)
            return content

        def apply_emphasis(content: str) -> str:
            # Replace **text** and *text* with HTML-like formatting
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            return content

        def format_section(title: str, content: str) -> str:
            # Format section with consistent spacing
            formatted_content = clean_text(content)
            formatted_content = process_document_references(formatted_content)
            formatted_content = format_lists(formatted_content)
            formatted_content = apply_emphasis(formatted_content)
            return f"# {title}\n\n{formatted_content}"

        # Process the content
        sections = []
        current_title = "Reply"
        current_content = []

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                if current_content:
                    sections.append(format_section(current_title, '\n'.join(current_content)))
                current_title = line.lstrip('#').strip()
                current_content = []
            elif line:
                current_content.append(line)

        if current_content:
            sections.append(format_section(current_title, '\n'.join(current_content)))

        return '\n\n'.join(sections)


class ImageBasedRAG:
    def __init__(self, model, processor, device, image_store, image_classifier):
        self.model = model
        self.processor = processor
        self.device = device
        self.image_store = image_store
        self.image_classifier = image_classifier
        self.labels = ["a technical image", "a non-technical image"]
        self.prompt_loader = PromptLoader()

    async def get_technical_context(self, image_data: Dict) -> Dict:
        """Extract and enrich technical context from image using GPT"""
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Convert image to base64 if it's not already
            if isinstance(image_data.get('image'), Image.Image):
                buffered = BytesIO()
                image_data['image'].save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                base64_image = image_data.get('image')

            # Create analysis prompt
            messages = [
                {
                    "role": "system",
                    "content": "Analyze this technical image from a UV water treatment perspective. Identify components, measurements, warning indicators, and maintenance-relevant features."
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

            # Get GPT analysis
            response = await client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=messages,
                max_tokens=150
            )

            # Extract insights from GPT response
            analysis = response.choices[0].message.content

            # Base context
            context = {
                "system_category": "UV Water Treatment System",
                "components_list": "",
                "documentation_refs": "",
                "maintenance_notes": "",
                "analysis": analysis
            }

            # Enhance with metadata if available
            if metadata := image_data.get('metadata', {}):
                if source := metadata.get('source', ''):
                    context['documentation_refs'] = f'"""{source}"""'
                context['maintenance_notes'] = metadata.get('maintenance_notes', '')

            return context

        except Exception as e:
            logging.error(f"Error in get_technical_context: {e}")
            return {
                "system_category": "Unknown",
                "components_list": "",
                "documentation_refs": "",
                "maintenance_notes": "",
                "analysis": ""
            }

    async def process_image_and_context(self, image_result: Dict, query_text: str, similarity: float) -> Tuple[
        Optional[Dict], float]:
        """Process a single image and get its context using templates"""
        metadata = image_result['metadata']
        image_id = (metadata.get('image', {}).get('id') or
                    metadata.get('content', {}).get('image_id'))

        if not image_id:
            return None, 0.0

        try:
            image, img_metadata = self.image_store.get_image(image_id)
            if not image:
                return None, 0.0

            # Check if it's a technical image
            predicted_label, confidence = self.image_classifier.classify(
                image=image,
                labels=self.labels,
            )

            if predicted_label != "a technical image" or confidence <= CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD:
                return None, 0.0

            # Calculate content similarity using CLIP embeddings
            image_input = self.processor(images=image, return_tensors="pt").to(self.device)
            image_embedding = self.model.get_image_features(**image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

            query_input = self.processor(
                text=[query_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            query_input = {k: v.to(self.device) for k, v in query_input.items()}
            query_embedding = self.model.get_text_features(**query_input)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

            # Calculate semantic similarity
            semantic_similarity = (query_embedding @ image_embedding.T).item()
            content_similarity = (semantic_similarity + similarity) / 2

            # Get base64 image data
            base64_image = self.image_store.get_base64(image_id)
            if not base64_image:
                return None, 0.0

            # Get technical context using templates
            technical_context = await self.get_technical_context(metadata)

            # Format image description
            image_description = self.prompt_loader.format_template(
                'image_description',
                source=str(metadata.get('path', '')),
                system_category=technical_context['system_category'],
                components_list=technical_context['components_list'],
                documentation_refs=technical_context['documentation_refs'],
                context_text=metadata.get('image', {}).get('context', ''),
                maintenance_notes=technical_context['maintenance_notes']
            )

            # Build query context
            query_context = self.prompt_loader.format_template(
                'image_query_with_context',
                query_text=query_text,
                image_context=image_description
            )

            image_data = {
                'image': base64_image,
                'image_id': image_id,
                'caption': metadata.get('image', {}).get('caption', ''),
                'context': query_context,
                'source': str(metadata.get('path', '')),
                'similarity': content_similarity,
                'semantic_similarity': semantic_similarity,
                'technical_confidence': confidence,
                'technical_details': technical_context
            }
            return image_data, content_similarity

        except Exception as e:
            logging.error(f"Error processing image {image_id}: {e}")
            return None, 0.0

    def validate_technical_image(self, image_data: Dict) -> bool:
        """Validate technical image data against template rules"""
        try:
            # Define validation rules directly based on prompts.yaml
            validation_rules = {
                'image': {
                    'max_size': 5 * 1024 * 1024,  # 5MB in bytes
                    'allowed_formats': ["jpg", "png", "svg"],
                    'min_dimensions': [100, 100],
                    'max_dimensions': [4000, 4000]
                }
            }

            # Check image size
            if len(image_data['image']) > validation_rules['image']['max_size']:
                return False

            # Check technical confidence
            if image_data.get('technical_confidence', 0) < CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD:
                return False

            # Validate required technical details
            tech_details = image_data.get('technical_details', {})
            if not tech_details.get('system_category') or not tech_details.get('components_list'):
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating technical image: {e}")
            return False


class RAGQueryServer:
    """
    Serves as the main backend for processing RAG-based queries, managing the index,
    metadata, and facilitating GPT-based chat interaction for text and image queries.

    Methods:
        determine_query_type(query_text): Determines the type of query based on input text.
        get_relevant_contexts(results, query_text): Filters the most relevant contexts and images.
        prepare_prompt(query_text, contexts, query_type, images): Constructs a GPT-ready prompt.
        process_text_query(query_text): Processes and retrieves a response for text queries.
    """

    def __init__(self):
        """
        Initializes the server by loading environment variables, setting up the
        OpenAI client, and preparing FAISS index and metadata for query processing.
        """
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.openai_api_key)
        self.model, self.processor, self.device = CLIP_init(CONFIG.CLIP_MODEL_NAME)
        self.prompt_loader = PromptLoader()

        # Try to load existing index or create new one
        try:
            self.index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            self.metadata = load_metadata(CONFIG.METADATA_PATH)
        except:
            logging.info("No existing index found, initializing new one")
            from utils.FAISS_utils import initialize_faiss_index, save_faiss_index, save_metadata
            self.index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
            self.metadata = []
            # Save empty index and metadata
            save_faiss_index(self.index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(self.metadata, CONFIG.METADATA_PATH)

        self.image_store = ImageStore()
        self.image_classifier = ImageClassifier(model=self.model, processor=self.processor, device=self.device)
        self.similarity_threshold = CONFIG.SIMILARITY_THRESHOLD
        self.formatter = EnhancedResponseFormatter()
        self.reset_chat()

        # Initialize AgentManager
        self.agent_manager = AgentManager(api_key=self.openai_api_key)

        logging.info(
            f"Server initialized with {len([m for m in self.metadata if m.get('type') == 'image'])} images in metadata")

    def determine_query_type(self, query_text: str) -> QueryType:
        query_lower = query_text.lower()
        return QueryType(
            is_overview="overview" in query_lower or "what is" in query_lower,
            is_technical="technical" in query_lower or "how" in query_lower,
            is_summary="summary" in query_lower or "brief" in query_lower
        )

    async def get_relevant_contexts(self, results: List[Dict], query_text: str) -> Tuple[List[str], List[Dict]]:
        """Get relevant contexts and images, filtering out non-technical images."""
        if not results or not results[0]:
            logging.info("No results found")
            return [], []

        logging.info(f"Processing query: {query_text}")
        logging.info(f"Found {len(results[0])} results")
        logging.info(f"Current thresholds: SIMILARITY={CONFIG.SIMILARITY_THRESHOLD}, "
                     f"IMAGE_SIMILARITY={CONFIG.IMAGE_SIMILARITY_THRESHOLD}, "
                     f"TECHNICAL_CONFIDENCE={CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD}")

        try:
            relevant_contexts = []
            relevant_images = []

            # Initialize ImageBasedRAG
            image_processor = ImageBasedRAG(
                model=self.model,
                processor=self.processor,
                device=self.device,
                image_store=self.image_store,
                image_classifier=self.image_classifier
            )

            for result in results[0]:
                metadata = result['metadata']
                similarity = 1 - (result['distance'] / 2)
                logging.info(
                    f"Processing result type: {metadata.get('type')} with similarity: {similarity:.4f} "
                    f"(raw distance: {result['distance']:.4f})")

                if metadata.get('type') == 'text-chunk':
                    if similarity > CONFIG.SIMILARITY_THRESHOLD:
                        if 'get_content' in metadata:
                            chunk_text = metadata['get_content']()
                            if chunk_text:
                                relevant_contexts.append(chunk_text.strip())
                                logging.info("Added text context")

                elif metadata.get('type') == 'image':
                    if similarity > CONFIG.IMAGE_SIMILARITY_THRESHOLD:
                        # Add await here
                        image_data, content_similarity = await image_processor.process_image_and_context(
                            image_result=result,
                            query_text=query_text,
                            similarity=similarity
                        )
                        if image_data:
                            relevant_images.append(image_data)
                            logging.info(f"Image content similarity score: {content_similarity:.4f}")

            relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
            logging.info(f"Final results: {len(relevant_contexts)} contexts, {len(relevant_images)} images")
            return relevant_contexts, relevant_images

        except Exception as e:
            logging.error(f"Error in get_relevant_contexts: {e}", exc_info=True)
            return [], []

    def get_image_data(self, image_id: str, metadata: Dict, similarity: float) -> Optional[Dict]:
        """Get image data with improved logging"""
        try:
            logging.info(f"Retrieving image data for ID: {image_id}")
            base64_image = self.image_store.get_base64(image_id)
            if not base64_image:
                logging.warning(f"Failed to get base64 image for ID: {image_id}")
                return None

            # Get all available metadata
            source_doc = metadata.get('source_doc', '')
            context = metadata.get('context', '')
            caption = metadata.get('caption', '')

            # Try to get additional metadata from content if available
            if isinstance(metadata.get('content'), dict):
                content = metadata['content']
                context = content.get('context', context)
                caption = content.get('caption', caption)

            image_data = {
                'image': base64_image,
                'image_id': image_id,
                'caption': caption,
                'context': context,
                'source': source_doc,
                'similarity': similarity
            }

            logging.info(f"Successfully retrieved image data for ID: {image_id}")
            return image_data
        except Exception as e:
            logging.error(f"Error getting image data for {image_id}: {e}")
            return None

    def prepare_prompt(self, query_text: str, contexts: List[str], query_type: QueryType, images: List[Dict]) -> list[
                                                                                                                     dict[
                                                                                                                         str, str]] | str:
        if not contexts and not images:
            # Use no-answer template if no relevant information
            return self.formatter.prompt_builder.build_no_answer_message(query_text)
        return self.formatter.prepare_prompt(query_text, contexts, query_type, images)

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        return self.formatter.prepare_messages(prompt)

    def get_images_from_referenced_documents(self, response_text: str) -> List[Dict]:
        """
        Get images from documents referenced in the OpenAI response text.
        Documents are referenced using double-double quotes, e.g., ""Document ID""

        Args:
            response_text: The text response from OpenAI containing document references

        Returns:
            List of relevant images with their metadata
        """
        try:
            # Extract document references using double-double quotes pattern
            referenced_docs = set()
            matches = re.finditer(r'""([^"]+)""', response_text)

            for match in matches:
                doc_ref = match.group(1).strip()
                # Remove any spaces from reference for matching
                doc_ref = doc_ref.replace(' ', '')
                referenced_docs.add(doc_ref)
                logging.info(f"Found document reference: {doc_ref}")

            if not referenced_docs:
                logging.info("No document references found in response")
                return []

            # Load image metadata
            with open(CONFIG.IMAGE_METADATA_PATH, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)

            # Get images from referenced documents
            relevant_images = []
            processed_ids = set()

            for image_id, img_data in image_metadata.items():
                source_doc = img_data.get('source_document', '')

                # Remove spaces and special characters from source_doc for matching
                clean_source = re.sub(r'[\s\-_.]', '', source_doc)

                # Check if any reference matches this source
                for doc_ref in referenced_docs:
                    if doc_ref in clean_source:
                        if image_id not in processed_ids:
                            try:
                                base64_image = self.image_store.get_base64(image_id)
                                if base64_image:
                                    image_info = {
                                        'image': base64_image,
                                        'image_id': image_id,
                                        'caption': img_data.get('caption', ''),
                                        'context': img_data.get('context', ''),
                                        'source': source_doc,
                                        'page_number': img_data.get('page_number'),
                                        'width': img_data.get('width'),
                                        'height': img_data.get('height')
                                    }
                                    relevant_images.append(image_info)
                                    processed_ids.add(image_id)
                                    logging.info(f"Added image {image_id} from document {source_doc}")
                            except Exception as e:
                                logging.error(f"Error processing image {image_id}: {e}")
                            break

            logging.info(f"Found {len(relevant_images)} images from {len(referenced_docs)} referenced documents")
            return relevant_images

        except Exception as e:
            logging.error(f"Error getting images from response: {e}")
            return []

    async def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> QueryResponse:
        try:
            logging.info(f"Processing query: {query_text}")

            # Check if any documents are indexed
            if not self.metadata:
                return QueryResponse(
                    text_response="No documents have been indexed yet. Please add some documents to the system first.",
                    images=[]
                )

            # Detect if query requires calculator agent
            agent_requirements = await self.agent_manager.detect_agent_requirements(query_text)

            # If it's a pure calculator query, handle it directly without RAG
            if agent_requirements.get('calculator'):
                try:
                    calc_results = await self.agent_manager.process_with_agents(
                        query_text,
                        agent_requirements
                    )
                    if calc_results and 'calculator' in calc_results:
                        logging.info(f"Calculator results: {calc_results['calculator']}")
                        text_response = await self.agent_manager.aggregate_responses(
                            "",  # Empty string since it's a pure calculation
                            calc_results
                        )
                        if text_response:
                            return QueryResponse(text_response=text_response, images=[])
                        else:
                            logging.error("Empty response from calculator aggregation")
                except Exception as e:
                    logging.error(f"Error in calculator processing: {e}")

            # If calculator processing failed, continue with RAG as fallback

            # Only proceed with RAG if it's not a pure calculator query
            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                top_k=top_k
            )

            # Handle empty RAG results
            if not results:
                logging.info("No relevant documents found")
                return QueryResponse(
                    text_response=self.formatter.prompt_builder.build_no_answer_message(query_text)[1]['content'],
                    images=[]
                )

            # Get contexts
            contexts, initial_images = await self.get_relevant_contexts(results, query_text)

            # Handle special cases
            if len(contexts) > 1 and "conflicting" in query_text.lower():
                logging.info("Detected potential conflict in contexts")
                conflicting_docs = [{"doc": context} for context in contexts]
                return QueryResponse(
                    text_response=self.formatter.prompt_builder.build_conflict_resolution_message(conflicting_docs)[1][
                        'content'],
                    images=[]
                )

            if "ambiguous" in query_text.lower():
                logging.info("Detected ambiguous query")
                return QueryResponse(
                    text_response=self.formatter.prompt_builder.build_ambiguity_message(query_text)[1]['content'],
                    images=[]
                )

            # Prepare and get response
            query_type = self.determine_query_type(query_text)
            prompt = self.prepare_prompt(query_text, contexts, query_type, [])

            # Add system message for concise responses
            if agent_requirements.get('calculator'):
                prompt.append({
                    "role": "system",
                    "content": "Provide only essential technical information. Avoid theoretical explanations."
                })

            response = openai_post_request(
                messages=self.prepare_messages(prompt),
                model_name=CONFIG.GPT_MODEL,
                max_tokens=CONFIG.DETAIL_MAX_TOKENS,
                temperature=CONFIG.TEMPERATURE if query_type.is_technical else 0.7,
                api_key=self.openai_api_key
            )

            text_response = response['choices'][0]['message']['content'].strip()

            # Get and process referenced images
            referenced_images = self.get_images_from_referenced_documents(text_response)

            # Combine both sets of images
            images = initial_images + referenced_images if initial_images else referenced_images

            if images:
                images = self.image_classifier.deduplicate(images, CONFIG.DEDUPLICATION_THRESHOLD)

            # Format response
            text_response = self.formatter.format_response(text_response)

            # Update chat history
            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({"role": "assistant", "content": text_response})

            return QueryResponse(text_response=text_response, images=images)

        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return QueryResponse(
                text_response="An error occurred while processing your query. Please try again.",
                images=[]
            )

    # In RAGQueryServer class

    async def _process_image_base(self, image_data: bytes, query_text: Optional[str] = None) -> dict:
        try:
            # Convert bytes to PIL Image and preprocess
            try:
                image = Image.open(BytesIO(image_data))
                # Handle image mode conversion
                if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    background.paste(image, mask=image.split()[-1])
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logging.error(f"Error preprocessing image: {str(e)}")
                raise HTTPException(status_code=400, detail="Error processing image format")

            # Convert image to base64
            try:
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=95)
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                logging.error(f"Error converting image to base64: {str(e)}")
                raise HTTPException(status_code=500, detail="Error preparing image for processing")

            # Get RAG contexts
            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                image_query=image,
                top_k=CONFIG.DEFAULT_TOP_K
            )
            contexts, related_images = await self.get_relevant_contexts(results, query_text or "")

            # Build structured context using templates
            technical_context = ""
            # In _process_image_base:
            if contexts:
                # For text contexts (documentation)
                technical_context = "Documentation Context:\n"
                for context in contexts:
                    technical_context += f"\nFrom Documentation: {context}\n"

                # For related images (if any)
                if related_images:
                    technical_context += "\nRelated Components in Documentation:\n"
                    for img in related_images:
                        if isinstance(img, dict):
                            img_context = self.prompt_loader.format_template(
                                'image_description',
                                source=img.get('source', 'Unknown'),
                                system_category="UV Water Treatment System",
                                components_list=str(img.get('technical_details', {}).get('components_list', '')),
                                documentation_refs=str(img.get('technical_details', {}).get('documentation_refs', '')),
                                context_text=img.get('context', ''),
                                maintenance_notes=str(img.get('technical_details', {}).get('maintenance_notes', ''))
                            )
                            technical_context += f"\n{img_context}\n"

            # Build query context
            query_context = self.prompt_loader.format_template(
                'image_query_with_context',
                query_text=query_text or "Analyze this technical image",
                image_context=technical_context
            )

            # Prepare messages using vision_assistant template
            messages = [
                {
                    "role": "system",
                    "content": self.prompt_loader.get_system_prompt('vision_assistant')
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                                {query_context}

                                IMPORTANT: If this component appears in our documentation, explicitly mention that and reference the document using double quotes.
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

            # Make API call with retries
            MAX_RETRIES = 3
            BASE_WAIT = 4
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=CONFIG.GPT_VISION_MODEL,
                        messages=messages,
                        max_tokens=CONFIG.VISION_MAX_TOKENS
                    )

                    answer = response.choices[0].message.content.strip()
                    formatter = EnhancedResponseFormatter()
                    formatted_answer = formatter.format_response(answer)

                    # Update chat history
                    self.chat_history.append({
                        "role": "user",
                        "content": f"[Image Query] {query_text or 'Analyze image'}"
                    })
                    self.chat_history.append({
                        "role": "assistant",
                        "content": formatted_answer
                    })

                    return {
                        "response": formatted_answer,
                        "related_documents": contexts,
                        "related_images": [
                            {
                                "caption": img.get("caption", ""),
                                "context": img.get("context", ""),
                                "source": img.get("source", ""),
                                "similarity": img.get("similarity", 0)
                            }
                            for img in related_images if isinstance(img, dict)
                        ]
                    }

                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        wait_time = BASE_WAIT * (2 ** attempt) + random.uniform(0, 1)
                        logging.warning(f"API error, retrying in {wait_time:.2f}s ({attempt + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logging.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                        raise

            raise HTTPException(
                status_code=500,
                detail=f"Failed after {MAX_RETRIES} attempts: {str(last_error)}"
            )

        except HTTPException as he:
            raise he
        except Exception as e:
            logging.error(f"Unexpected error in image processing: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> dict:
        """Process an image query from raw bytes."""
        return await self._process_image_base(image_data, query_text)

    def reset_chat(self):
        self.chat_history = []
        logging.info("Chat history has been reset")
        return {"status": "success", "message": "Chat history cleared"}

    def get_chat_history(self):
        return self.chat_history


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logging.info("Starting RAG Query Server...")
    os.makedirs("static", exist_ok=True)
    os.makedirs(CONFIG.STORED_IMAGES_PATH, exist_ok=True)
    logging.info("Directory structure verified")

    yield  # The application runs during this phase

    # Shutdown logic
    logging.info("Shutting down RAG Query Server...")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan, title="Atlantium RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize server
server = RAGQueryServer()

# Serve static files
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    with open("static/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content, headers=headers)


# Add cache prevention middleware
@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.post("/chat/reset")
async def reset_chat():
    return server.reset_chat()


@app.post("/query/text")
async def text_query(query: str = Form(...)):
    """
    Handles text-based queries by retrieving relevant contexts and generating a response.
    """
    try:
        # Process the query
        response = await server.process_text_query(query)

        # Create proper response structure
        response_data = {
            "status": "success",
            "response": {
                "text_response": response.text_response,
                "images": response.images
            }
        }

        # Log the response
        logging.info(f"Sending response with {len(response.images)} images")

        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"Error processing text query: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "response": {
                    "text_response": "Sorry, there was an error processing your request.",
                    "images": []
                }
            },
            status_code=500
        )


@app.post("/query/image")
async def image_query(
        image: UploadFile = File(...),
        query: Optional[str] = Form(None)
):
    try:
        if query is None:
            query = "Please analyze this image and provide a detailed description based on our technical documentation."

        # Initial validation
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found")

        # Check file size and process image
        contents = await image.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File size too large")

        # Process using the base method
        return await server._process_image_base(contents, query)

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/document")
async def upload_document(file: UploadFile, folder: str = Form("")):
    """
    Handles uploading of documents to a specified folder on the server.

    Args:
        file (UploadFile): The file to be uploaded.
        folder (str): The target folder for the upload.

    Returns:
        dict: A success status and the relative path of the saved file.
    """
    try:
        # Clean and decode the folder path
        clean_folder = clean_path(folder)

        # Create full target directory path
        target_dir = CONFIG.RAW_DOCUMENTS_PATH
        if clean_folder:
            target_dir = target_dir / clean_folder
            # Ensure target directory exists and is within RAW_DOCUMENTS_PATH
            if not str(target_dir).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
                raise HTTPException(status_code=403, detail="Invalid folder path")
            target_dir.mkdir(parents=True, exist_ok=True)

        # Clean and decode filename
        sanitized_filename = clean_path(file.filename)
        if not sanitized_filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Validate file extension
        if not any(sanitized_filename.lower().endswith(ext)
                   for ext in CONFIG.SUPPORTED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Create full path for the file
        dest_path = target_dir / sanitized_filename

        # Avoid overwriting existing files
        if dest_path.exists():
            raise HTTPException(status_code=409, detail="File already exists")

        # Save the file safely
        try:
            with open(dest_path, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            logging.info(f"File saved to {dest_path}")
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail="Error saving file")
        finally:
            await file.close()

        # Return relative path without URL encoding
        relative_path = str(dest_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH))
        return {
            "status": "success",
            "path": relative_path
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def update_processed_files(doc_paths):
    """Update the list of successfully processed files"""
    processed_files_path = Path("processed_files.json")
    try:
        if processed_files_path.exists():
            with open(processed_files_path, 'r') as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()

        # Add new files
        processed_files.update([str(unquote(path)) for path in doc_paths])

        # Save updated list
        with open(processed_files_path, 'w') as f:
            json.dump(list(processed_files), f)

    except Exception as e:
        logging.error(f"Error updating processed files list: {e}")


def check_processing_status():
    """Check if all necessary files and data exist after processing"""
    logger = logging.getLogger(__name__)
    try:
        # Check required paths
        if not CONFIG.METADATA_PATH.exists():
            logger.error("Metadata file not found")
            return False, "Metadata file not found"

        if not CONFIG.FAISS_INDEX_PATH.exists():
            logger.error("FAISS index not found")
            return False, "FAISS index not found"

        # Check metadata content with explicit UTF-8 encoding
        try:
            with open(CONFIG.METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if not metadata:
                    logger.error("Empty metadata file")
                    return False, "Empty metadata file"
        except json.JSONDecodeError as e:
            logger.error(f"Invalid metadata file: {e}")
            return False, "Invalid metadata file format"
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in metadata file: {e}")
            return False, "Encoding error in metadata file"

        # Check index
        try:
            index = faiss.read_index(str(CONFIG.FAISS_INDEX_PATH))
            if index.ntotal == 0:
                logger.error("Empty FAISS index")
                return False, "Empty FAISS index"
        except Exception as e:
            logger.error(f"Error reading FAISS index: {e}")
            return False, f"Error reading FAISS index: {str(e)}"

        logger.info("All processing checks passed successfully")
        return True, "Processing completed successfully"

    except Exception as e:
        logger.error(f"Error checking processing status: {str(e)}")
        return False, f"Error checking processing status: {str(e)}"


@app.post("/process/documents")
@app.post("/process/documents")
async def process_documents():
    """
    Process documents asynchronously while maintaining metadata persistence.
    Returns a status response indicating success or failure.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting document processing...")

        # Load existing metadata before processing
        existing_metadata = []
        if CONFIG.METADATA_PATH.exists():
            try:
                with open(CONFIG.METADATA_PATH, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                logger.info(f"Loaded {len(existing_metadata)} existing metadata entries")
            except Exception as e:
                logger.warning(f"Could not load existing metadata: {e}")

        # Run RAG_processor.py with proper encoding environment variable
        process = subprocess.Popen(
            [sys.executable, 'RAG_processor.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8"
            }
        )

        stdout, stderr = process.communicate()

        # Process and log stdout
        if stdout:
            for line in stdout.splitlines():
                if 'ERROR' in line:
                    logger.error(line)
                else:
                    logger.info(line)

        # Process and log stderr
        if stderr:
            for line in stderr.splitlines():
                if 'ERROR' in line:
                    logger.error(f"Processing error: {line}")
                else:
                    logger.info(line)

        # Check return code
        if process.returncode != 0:
            error_msg = f"Process failed with code {process.returncode}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Verify the results
        success, message = check_processing_status()
        if not success:
            logger.error(f"Processing verification failed: {message}")
            raise HTTPException(status_code=500, detail=message)

        # Merge new metadata with existing metadata
        try:
            # Load newly processed metadata
            with open(CONFIG.METADATA_PATH, 'r', encoding='utf-8') as f:
                new_metadata = json.load(f)

            # Helper function to generate unique key for metadata entry
            def get_entry_key(entry):
                if entry.get('type') == 'image':
                    return f"image_{entry.get('image', {}).get('id')}"
                elif entry.get('type') == 'text-chunk':
                    return f"chunk_{entry.get('path')}_{entry.get('chunk')}"
                else:
                    content_str = json.dumps(entry.get('content', {}), sort_keys=True)
                    return f"other_{hashlib.md5(content_str.encode()).hexdigest()}"

            # Use dictionary for O(1) lookups
            merged_metadata = {}

            # Add existing metadata first
            for entry in existing_metadata:
                entry_key = get_entry_key(entry)
                merged_metadata[entry_key] = entry

            # Add new metadata
            for entry in new_metadata:
                entry_key = get_entry_key(entry)
                if entry_key not in merged_metadata:
                    merged_metadata[entry_key] = entry

            # Convert back to list
            final_metadata = list(merged_metadata.values())

            # Save merged metadata
            with open(CONFIG.METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully merged metadata: {len(final_metadata)} total entries")

            # Reload the server's index and metadata
            server.index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            server.metadata = final_metadata

        except Exception as e:
            logger.error(f"Error merging metadata: {e}")
            raise HTTPException(status_code=500, detail="Failed to merge metadata")

        # Verify the final state
        try:
            if not server.index or not server.metadata:
                raise ValueError("Index or metadata is empty after processing")

            logger.info(f"Final verification: {len(server.metadata)} metadata entries, "
                        f"{server.index.ntotal} vectors in index")
        except Exception as e:
            logger.error(f"Final verification failed: {e}")
            raise HTTPException(status_code=500, detail="Final verification failed")

        logger.info("Document processing completed successfully")
        return {"status": "success"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process_documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get/documents")
async def get_documents(path: str = ""):
    """Get list of documents and folders with metadata recursively"""
    logger = logging.getLogger(__name__)
    try:
        # Get total count from processed_files.json
        processed_files_path = Path("processed_files.json")
        total_documents = 0
        if processed_files_path.exists():
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                total_documents = len(json.load(f))

        # Decode URL-encoded path and clean it
        decoded_path = unquote(path)
        clean_folder = clean_path(decoded_path)
        current_path = CONFIG.RAW_DOCUMENTS_PATH / clean_folder if clean_folder else CONFIG.RAW_DOCUMENTS_PATH

        if not current_path.exists() or not current_path.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")

        if not str(current_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")

        folders = []
        files = []

        # Get directories and files in current path
        try:
            for item in current_path.iterdir():
                try:
                    stat = item.stat()
                    rel_path = item.relative_to(CONFIG.RAW_DOCUMENTS_PATH)

                    if item.is_dir():
                        folders.append({
                            "name": item.name,
                            "path": str(rel_path),  # No need to URL encode here
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                    elif item.is_file() and any(item.name.lower().endswith(ext) for ext in CONFIG.SUPPORTED_EXTENSIONS):
                        files.append({
                            "name": item.name,
                            "path": str(rel_path),  # No need to URL encode here
                            "type": item.suffix[1:].upper(),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    continue

            # Sort folders and files alphabetically
            folders.sort(key=lambda x: x['name'].lower())
            files.sort(key=lambda x: x['name'].lower())

            logger.info(f"Found {len(folders)} folders and {len(files)} files in {current_path}")
            return {
                "current_path": str(clean_folder),  # Use unencoded path for display
                "folders": folders,
                "files": files,
                "total_processed": total_documents  # Added total processed documents count
            }

        except Exception as e:
            logger.error(f"Error reading directory {current_path}: {e}")
            raise HTTPException(status_code=500, detail="Error reading directory")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    try:
        # Load list of processed documents
        processed_files_path = Path("processed_files.json")
        if processed_files_path.exists():
            with open(processed_files_path, 'r') as f:
                documents = json.load(f)

            # Get file details
            doc_details = []
            for doc_path in documents:
                path = Path(doc_path)
                if path.exists():
                    stats = path.stat()
                    doc_details.append({
                        "name": path.name,
                        "size": stats.st_size,
                        "modified": stats.st_mtime,
                        "type": path.suffix[1:].upper()
                    })

            return {"documents": doc_details}
        return {"documents": []}
    except Exception as e:
        logging.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    try:
        # Sanitize and validate the file path
        sanitized_path = clean_path(file_path)
        full_path = CONFIG.RAW_DOCUMENTS_PATH / sanitized_path

        # Ensure the file exists and is accessible
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(full_path)

    except Exception as e:
        logging.error(f"Error serving file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/open/document")
async def open_document(path: str = Body(..., embed=True)):
    try:
        sanitized_path = clean_path(path)
        full_path = CONFIG.RAW_DOCUMENTS_PATH / sanitized_path

        # Ensure the file exists and is accessible
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Return the absolute URL to access the file via `/files`
        # Replace with your actual server URL if needed
        file_url = f"/files/{sanitized_path}"
        return {"status": "success", "url": file_url}

    except Exception as e:
        logging.error(f"Error generating file URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/document")
async def download_document(path: str):
    """Download a document."""
    try:
        # Clean and validate the path
        clean_file_path = clean_path(path)
        full_path = CONFIG.RAW_DOCUMENTS_PATH / clean_file_path

        # Security check
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=full_path,
            filename=full_path.name,
            media_type="application/octet-stream"
        )

    except Exception as e:
        logging.error(f"Error downloading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete/document")
async def delete_document(path: str):
    """Delete a document and remove it from RAG."""
    try:
        # Properly decode URL-encoded path
        clean_file_path = clean_path(path)
        full_path = CONFIG.RAW_DOCUMENTS_PATH / clean_file_path

        logging.info(f"Received request to delete document: {full_path}")

        # Security check
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            logging.warning(f"Access denied: Path {full_path} is outside of allowed directory")
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists():
            logging.warning(f"File not found: {full_path}")
            raise HTTPException(status_code=404, detail="File not found")

        # Remove from RAG first
        success, message = remove_document_from_rag(full_path)
        if not success:
            logging.error(f"Failed to remove document from RAG: {message}")
            raise HTTPException(status_code=500, detail=message)

        # Delete the file
        try:
            os.remove(full_path)
            logging.info(f"Successfully deleted file: {full_path}")
        except Exception as e:
            logging.error(f"Failed to delete file {full_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

        return {"status": "success", "message": "File deleted and removed from RAG"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folder/create")
async def create_new_folder(
        parent_path: str = Body(..., embed=True),
        folder_name: str = Body(..., embed=True)
):
    """Create a new folder."""
    try:
        clean_parent_path = clean_path(parent_path)
        full_parent_path = CONFIG.RAW_DOCUMENTS_PATH / clean_parent_path

        # Security check
        if not str(full_parent_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")

        success, message = create_folder(full_parent_path, folder_name)
        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {"status": "success", "message": message}

    except Exception as e:
        logging.error(f"Error creating folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folder/delete")
async def delete_folder(path: str):
    """Delete a folder and remove its contents from RAG."""
    try:
        clean_folder_path = clean_path(path)
        full_path = CONFIG.RAW_DOCUMENTS_PATH / clean_folder_path

        # Security check
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        success, message, errors = delete_folder_from_rag(full_path)
        if not success:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": message,
                    "errors": errors
                }
            )

        return {
            "status": "success",
            "message": message,
            "errors": errors if errors else None
        }

    except Exception as e:
        logging.error(f"Error deleting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/folder/rename")
async def rename_folder(
        path: str = Body(..., embed=True),
        new_name: str = Body(..., embed=True)
):
    """Rename a folder and update RAG references."""
    try:
        clean_folder_path = clean_path(path)
        full_path = CONFIG.RAW_DOCUMENTS_PATH / clean_folder_path

        # Security check
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        # Validate new name
        valid, message = validate_folder_name(new_name)
        if not valid:
            raise HTTPException(status_code=400, detail=message)

        # Create new path
        new_path = full_path.parent / new_name
        if new_path.exists():
            raise HTTPException(status_code=400, detail="A folder with this name already exists")

        success, message = rename_folder_in_rag(full_path, new_path)
        if not success:
            raise HTTPException(status_code=500, detail=message)

        return {"status": "success", "message": message}

    except Exception as e:
        logging.error(f"Error renaming folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rescan")
async def rescan_documents_endpoint():
    """Rescan documents and update RAG system."""
    try:
        success, message = rescan_documents(CONFIG)

        if not success:
            raise HTTPException(status_code=500, detail=message)

        return {"status": "success", "message": message}

    except Exception as e:
        logging.error(f"Error during rescan: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/chat/history")
async def get_chat_history():
    """
    Retrieves the current chat history for the session.

    Returns:
        dict: Contains a list of chat history messages in chronological order.
    """
    history = server.get_chat_history()
    return {"history": history}


@app.get('/favicon.ico', include_in_schema=False)
@app.get('/favicon.png', include_in_schema=False)
async def favicon():
    favicon_path = Path('static/favicon.png')  # Create this file or adjust the path
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return {'status_code': 404}


# Run server configuration
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=CONFIG.SERVER_PORT,
        log_level="info"
    )
