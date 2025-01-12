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

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
from contextlib import asynccontextmanager
from dataclasses import field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import unquote

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
from models.image_processor import ImageProcessor, ImageClassifier
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
from utils.img_utils import ImageProcessor as ImageUtils
from models.agents.agent_manager import AgentManager
from models.agents.websearch_agent import WebSearchAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

# Disable the UserWarning from Flash-Attention (GPU capabilities >8.0)
warnings.simplefilter("ignore", UserWarning)

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
logger = logging.getLogger(__name__)


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
    is_general: bool = True


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

    @staticmethod
    def format_response(content: str, chunk_metadata: List[Dict] = None) -> str:
        """
        Format the response text with proper styling and document references.

        Args:
            content (str): The raw content to format
            chunk_metadata (List[Dict], optional): Metadata from text chunks containing document information

        Returns:
            str: Formatted content with proper styling and document references
        """

        def process_document_references(text: str) -> str:
            """
            Process document references using metadata from text chunks and processed files.

            Args:
                text (str): The text containing document references
            """
            try:
                # Load processed files for fallback
                processed_files_path = CONFIG.PROCESSED_FILES_PATH
                if not processed_files_path.exists():
                    logging.warning("processed_files file not found")
                    return text

                with processed_files_path.open('r', encoding='utf-8') as f:
                    processed_files = json.load(f)

                # Create document mapping from metadata
                doc_mapping = {}

                # First, build mapping from chunk metadata if available
                if chunk_metadata:
                    for meta in chunk_metadata:
                        if isinstance(meta, dict) and 'path' in meta:
                            # Get the full path
                            full_path = str(meta['path'])
                            # Extract filename
                            filename = os.path.basename(full_path)
                            # Extract unique document ID
                            match = re.match(r'^([A-Za-z0-9]+)-', filename)
                            if match:
                                doc_id = match.group(1)
                                # Store both ID and full path
                                doc_mapping[doc_id] = {
                                    'filename': filename,
                                    'path': full_path,
                                    'relative_path': full_path.split('Raw Documents/')[-1]
                                    if 'Raw Documents/' in full_path
                                    else full_path
                                }
                                logging.debug(f"Added document mapping from metadata: {doc_id} -> {filename}")

                # Fallback to processed_files for any missing mappings
                for file_path in processed_files:
                    if not isinstance(file_path, str):
                        continue

                    norm_path = file_path.replace('\\', '/')
                    filename = os.path.basename(norm_path)
                    match = re.match(r'^([A-Za-z0-9]+)-', filename)
                    if match:
                        doc_id = match.group(1)
                        if doc_id not in doc_mapping:
                            relative_path = (norm_path.split('Raw Documents/')[1]
                                             if 'Raw Documents/' in norm_path
                                             else norm_path)
                            doc_mapping[doc_id] = {
                                'filename': filename,
                                'path': norm_path,
                                'relative_path': relative_path
                            }
                            logging.debug(f"Added document mapping from processed files: {doc_id} -> {filename}")

                def replacement(ref_match):
                    doc_ref = ref_match.group(1).strip()
                    # Find the full path for this document ID
                    fullpath = None

                    for filepath in processed_files:
                        normpath = filepath.replace('\\', '/')
                        # Get filename without path
                        file_name = os.path.basename(normpath)
                        # Check if filename starts with the document ID
                        if file_name.startswith(f"{doc_ref}-"):
                            if 'Raw Documents/' in normpath:
                                fullpath = normpath.split('Raw Documents/')[1]
                            else:
                                fullpath = normpath
                            break

                    if fullpath:
                        # Use the full filename in the display text
                        display_text = os.path.basename(fullpath)
                        safe_path = fullpath.replace("'", "\\'")
                        return f'<a href="javascript:void(0)" onclick="openDocument(\'{safe_path}\')" class="doc-link">{display_text}</a>'

                    # Return the original reference if no match found
                    logging.debug(f"No match found for document: {doc_ref}")
                    return doc_ref

                # Find and replace document references
                pattern = r'\[ref](.*?)\[/ref]'
                text = re.sub(pattern, replacement, text)
                return text

            except Exception as err:
                logging.error(f"Error processing document references: {err}", exc_info=True)
                return text

        def clean_text(text: str) -> str:
            """Clean up excess whitespace while preserving structure."""
            text = re.sub(r'\s*\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()

        def format_lists(list_content: str) -> str:
            """Format lists with proper spacing and indentation."""
            # Add <br> before valid numbered list items
            list_content = re.sub(r'([^\n])\s*(\d+\.\s+(?=[A-Za-z]))', r'\1<br>\2', list_content)
            # Add <br> before headers
            list_content = re.sub(r'([^\n])\s*(#{2,3}\s+)', r'\1<br>\2', list_content)
            # Format bullet points with proper indentation
            list_content = re.sub(r'(?m)^[•\-]\s*', r'  • ', list_content)
            # Remove bullet points from bold text items with bullets
            list_content = re.sub(r'\*\*\s*•\s*', r'• ', list_content)
            # Format numbered lists with proper indentation
            list_content = re.sub(r'(?m)^(\d+\.\s+)(\*\*.*?\*\*)', r'    \1\2', list_content)
            # Ensure line breaks between list items
            list_content = re.sub(r'(?<!<br>)(\d+\.\s+)(\*\*.*?\*\*)', r'<br>\1\2', list_content)
            return list_content

        def apply_emphasis(emphasis_content: str) -> str:
            """Apply emphasis formatting while preserving document references."""
            # First, temporarily protect [ref] tags
            emphasis_content = re.sub(r'\[ref](.*?)\[/ref]', r'PRESERVED_REF{\1}PRESERVED_REF', emphasis_content)
            # Replace **text** and *text* with HTML-like formatting
            emphasis_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', emphasis_content)
            emphasis_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', emphasis_content)
            # Restore [ref] tags
            emphasis_content = re.sub(r'PRESERVED_REF{(.*?)}PRESERVED_REF', r'[ref]\1[/ref]', emphasis_content)
            return emphasis_content

        def format_section(title: str, section_content: str) -> str:
            """Format a section with all necessary formatting applied."""
            try:
                formatted_content = clean_text(section_content)
                formatted_content = format_lists(formatted_content)
                formatted_content = apply_emphasis(formatted_content)
                formatted_content = process_document_references(formatted_content)
                return f"# {title}\n\n{formatted_content}"
            except Exception as err:
                logging.error(f"Error formatting section '{title}': {err}", exc_info=True)
                return f"# {title}\n\n{section_content}"

        try:
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

        except Exception as e:
            logging.error(f"Error formatting response: {e}", exc_info=True)
            return content  # Return original content if formatting fails


class RAGQueryServer:
    """
    Serves as the main backend for processing RAG-based queries, managing the index,
    metadata, and facilitating GPT-based chat interaction for text and image queries.
    """

    def __init__(self):
        # Initialize environment and API
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)

        # Initialize CLIP model
        self.model, self.device = CLIP_init(CONFIG.CLIP_MODEL_NAME)

        self.prompt_loader = PromptLoader()
        self.formatter = EnhancedResponseFormatter()
        self.agent_manager = AgentManager(api_key=self.openai_api_key)

        # Initialize components
        self.image_processor = ImageProcessor(
            openai_client=self.client,
            model=self.model,
            device=self.device,
            formatter=self.formatter
        )
        self.image_classifier = ImageClassifier(
            model=self.model,
            device=self.device,
        )

        # Load processed files from metadata
        self.processed_files = self._load_processed_files()

        # Initialize index and chat history
        self._initialize_index()
        self.chat_history = []

        logging.info(
            f"Server initialized with {len([m for m in self.metadata if m.get('type') == 'image'])} images in metadata"
        )

    @staticmethod
    def _load_processed_files() -> List[str]:
        """Load the list of processed files from metadata."""
        try:
            metadata_path = CONFIG.PROCESSED_FILES_PATH  # Make sure this is defined in your config
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logging.warning(f"No processed files metadata found at {metadata_path}")
                return []
        except Exception as e:
            logging.error(f"Error loading processed files metadata: {e}")
            return []

    def _initialize_index(self):
        """Initialize or load existing FAISS index and metadata."""
        try:
            self.index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            self.metadata = load_metadata(CONFIG.METADATA_PATH)
        except Exception as e:
            logging.info(f"No existing index found, initializing new one: {e}")
            from utils.FAISS_utils import initialize_faiss_index, save_faiss_index, save_metadata
            self.index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
            self.metadata = []
            save_faiss_index(self.index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(self.metadata, CONFIG.METADATA_PATH)

    async def determine_query_type(self, query_text: str) -> QueryType:
        """Determine the type of query based on semantic analysis."""
        try:
            # GPT prompt messages
            messages = [
                {
                    "role": "system",
                    "content": """
                        You are the specialist, able to classify queries based on their content.
                        Analyze queries to determine their type. Consider:
                        - Query domain (technical question, technical summary, technical overview or general knowledge)
                        - Reply with the following types: overview, technical, summary, general
                        - Choose ONLY one, most relevant type based on the query
                        - All non-technical questions shall be classified as "general"
                        Analyze the full semantic meaning of the query."""
                },
                {
                    "role": "user",
                    "content": f"Classify this query: {query_text}"
                }
            ]

            response = openai_post_request(
                messages=messages,
                model_name=CONFIG.BASE_LLM_MODEL,
                max_tokens=CONFIG.SUMMARY_MAX_TOKENS,
                temperature=0,
                api_key=self.openai_api_key
            )

            # Extract and parse response content
            classification = response['choices'][0]['message']['content'].strip()

            # Default is_technical:
            if classification == "overview":
                return QueryType(
                    is_overview=True,
                    is_technical=False,
                    is_summary=False,
                    is_general=False
                )
            elif classification == "summary":
                return QueryType(
                    is_overview=False,
                    is_technical=False,
                    is_summary=True,
                    is_general=False
                )
            elif classification == "general":
                return QueryType(
                    is_overview=False,
                    is_technical=False,
                    is_summary=False,
                    is_general=True
                )
            else:
                return QueryType(
                    is_overview=False,
                    is_technical=True,
                    is_summary=False,
                    is_general=False
                )

        except Exception as e:
            logging.error(f"Error determining query type: {e}")
            # Fallback to default query type
            return QueryType(
                is_overview=False,
                is_technical=False,
                is_summary=False,
                is_general=True  # Default fallback assumption
            )

    async def _process_image_result(self, result: Dict, query_text: str) -> List[Dict]:
        """Process individual image search result to extract basic image information."""
        try:
            metadata = result['metadata']
            image_id = (metadata.get('image', {}).get('id') or
                        metadata.get('content', {}).get('image_id'))

            if not image_id:
                return []

            # Get base64 image directly
            base64_image = self.image_processor.image_store.get_base64(image_id)
            if not base64_image:
                return []

            return [{
                'image': base64_image,
                'image_id': image_id,
                'caption': metadata.get('image', {}).get('caption', ''),
                'context': query_text,
                'source': str(metadata.get('path', '')),
                'similarity': 1.0,  # Default value since we're not calculating similarity
                'semantic_similarity': 1.0,  # Default value since we're not calculating similarity
                'technical_details': {}  # Empty dict since we're not analyzing technical context
            }]

        except Exception as e:
            logging.error(f"Error processing image result: {e}")
            return []

    async def _calculate_image_similarity(self, image: Image.Image, query_text: str, distance: float) -> Dict:
        """Calculate image-text similarity using CLIP embeddings."""
        return await ImageUtils.calculate_image_similarity(
            image=image,
            query_text=query_text,
            distance=distance,
            model=self.model,
            device=self.device
        )

    async def _get_referenced_images(self, response_text: str) -> List[Dict]:
        """Get images from documents referenced in the response text."""
        # Meanwile disabled
        try:
            # Extract document references using pattern
            referenced_docs = set()
            matches = re.finditer(r'\[ref](.*?)\[/ref]', response_text)

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
            image_candidates = []
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
                                base64_image = self.image_processor.image_store.get_base64(image_id)
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
                                    image_candidates.append(image_info)
                                    processed_ids.add(image_id)
                                    logging.info(f"Added image {image_id} from document {source_doc}")
                            except Exception as e:
                                logging.error(f"Error processing image {image_id}: {e}")
                            break

            if not image_candidates:
                logging.info("No images found in referenced documents")
                return []

            try:
                # Use ImageClassifier to rank images by relevance
                if hasattr(self, 'image_classifier'):
                    ranked_images = self.image_classifier.rank_images_by_relevance(
                        image_store=self.image_processor.image_store,
                        images=image_candidates,
                        query_text=response_text
                    )
                    logging.info(f"Selected {len(ranked_images)} most relevant images")
                    return ranked_images[:13]  # Limit to 12 most relevant images
                else:
                    logging.info("No image classifier available, returning unranked images")
                    return image_candidates[:13]  # Return first 12 images if no classifier

            except Exception as e:
                logging.error(f"Error ranking images: {e}", exc_info=True)
                return image_candidates[:13]  # Fallback to first 12 images if ranking fails

        except Exception as e:
            logging.error(f"Error getting images from response: {e}")
            return []

    @staticmethod
    def _extract_document_references(chunk_metadata: List[Dict]) -> List[str]:
        """Extract unique document IDs from chunk metadata."""
        doc_refs = set()
        for meta in chunk_metadata:
            if isinstance(meta, dict) and 'path' in meta:
                filename = os.path.basename(meta['path'])
                match = re.match(r'^([A-Za-z0-9]+)-', filename)
                if match:
                    doc_refs.add(match.group(1))
        return sorted(list(doc_refs))

    async def process_text_query(self, query_text: str) -> QueryResponse:
        """Process a text query and return response with relevant images."""
        try:
            if not self.metadata:
                return QueryResponse(
                    text_response="No documents have been indexed yet. Please add some documents to the system first.",
                    images=[]
                )

            # Detect if query requires calculator agent
            agent_requirements = await self.agent_manager.detect_agent_requirements(query_text)

            # If it's a calculator query, handle it with proper error reporting
            if agent_requirements.get('calculator'):
                try:
                    calc_results = await self.agent_manager.process_with_agents(
                        query_text,
                        agent_requirements
                    )

                    if calc_results and 'calculator' in calc_results:
                        # Always use aggregate_responses for formatting
                        text_response = await self.agent_manager.aggregate_responses(
                            "",  # Empty string since it's a pure calculation
                            calc_results
                        )
                        # Update the history and return
                        self.update_chat_history(query_text, text_response)
                        return QueryResponse(text_response=text_response, images=[])

                except Exception as e:
                    logging.error(f"Error in calculator processing: {e}")
                    return QueryResponse(
                        text_response=f"Error in calculation: {str(e)}",
                        images=[]
                    )

            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                device=self.device,
                text_query=query_text,
            )

            if not results:
                return QueryResponse(
                    text_response="No relevant information found. Please try again.",
                    images=[]
                )

            contexts = []
            initial_images = []
            chunk_metadata = []

            if results and results[0]:
                for result in results[0]:
                    metadata = result['metadata']
                    if metadata.get('type') == 'text-chunk' and 'get_content' in metadata:
                        contexts.append(metadata['get_content']().strip())
                        chunk_metadata.append(metadata)
                    elif metadata.get('type') == 'image':
                        processed_images = await self._process_image_result(result, query_text)
                        if processed_images:
                            initial_images.extend(processed_images)

            # Extract document references from chunk metadata
            available_refs = self._extract_document_references(chunk_metadata)

            query_type = await self.determine_query_type(query_text)

            # if no contexts OR general question: return websearch results
            if not contexts or query_type.is_general:
                contexts = self._create_no_results_response(query_text)
                initial_images = []  # Drop the images, found in RAG, they are unrelated to the websearch

            # Get chat history with proper formatting
            formatted_history = self.get_chat_history()

            formatted_prompt = self.formatter.prompt_builder.build_chat_prompt(
                query_text=query_text,
                contexts=contexts,
                images=initial_images,
                chat_history=formatted_history,
                available_refs=available_refs,
                is_technical=query_type.is_technical,
                is_summary=query_type.is_summary,
                is_overview=query_type.is_overview
            )

            # Prepare messages for OpenAI
            messages = self.formatter.prompt_builder.build_messages(formatted_prompt)

            # Get response from OpenAI
            response = openai_post_request(
                messages=messages,
                model_name=CONFIG.BASE_LLM_MODEL,
                max_tokens=CONFIG.DETAIL_MAX_TOKENS,  # Detail? Why not general?
                temperature=CONFIG.TEMPERATURE,
                api_key=self.openai_api_key
            )

            # Get the raw response text
            text_response = response['choices'][0]['message']['content'].strip()

            # Get images from referenced documents using the image processor
            # referenced_images = await self._get_referenced_images(text_response)
            # Meanwhile disabling this feature
            referenced_images = []

            # Combine and deduplicate images using the image processor
            all_images = initial_images + referenced_images if initial_images else referenced_images
            if all_images:
                all_images = self.image_processor.image_classifier.deduplicate(all_images,
                                                                               CONFIG.DEDUPLICATION_THRESHOLD)

            # Format the response text
            formatted_response = self.formatter.format_response(text_response, chunk_metadata)

            # Update chat history
            self.update_chat_history(query_text, formatted_response)

            # Create and return the final response
            final_response = QueryResponse(
                text_response=formatted_response,
                images=all_images
            )

            return final_response

        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return QueryResponse(
                text_response="An error occurred while processing your query. Please try again.",
                images=[]
            )

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> Dict:
        """Process image query using ImageProcessor."""
        try:
            result = await self.image_processor.process_image_query(image_data, query_text)
            if result.get('is_technical', False):
                self.update_chat_history(
                    f"[Image Query] {query_text or 'Analyze image'}",
                    result.get('response', '')
                )
            return result
        except Exception as e:
            logging.error(f"Error in process_image_query: {e}")
            raise

    def reset_chat(self):
        """Reset chat history."""
        self.chat_history = []
        logging.info("Chat history has been reset")
        return {"status": "success", "message": "Chat history cleared"}

    def get_chat_history(self):
        """Get current chat history."""
        return self.chat_history

    def update_chat_history(self, query: str, response: str):
        """Update chat history with new query and response up to N times."""
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": response})
        # Pop last N records:
        if len(self.chat_history) > 2 * CONFIG.MAX_CHAT_HISTORY:
            self.chat_history = self.chat_history[-(2 * CONFIG.MAX_CHAT_HISTORY):]

    async def _handle_calculator_query(self, query_text: str) -> Optional[QueryResponse]:
        """Handle calculator-specific queries."""
        try:
            agent_requirements = await self.agent_manager.detect_agent_requirements(query_text)
            if agent_requirements.get('calculator'):
                calc_results = await self.agent_manager.process_with_agents(query_text, agent_requirements)
                if calc_results and 'calculator' in calc_results:
                    text_response = await self.agent_manager.aggregate_responses("", calc_results)
                    if text_response:
                        return QueryResponse(text_response=text_response, images=[])
        except Exception as e:
            logging.error(f"Error in calculator processing: {e}")
        return None

    @staticmethod
    def _create_no_results_response(query_text: str) -> str:
        """Create response for when no context is found by web-search agent"""
        websearch = WebSearchAgent(model=CONFIG.WEBSEARCH_MODEL, max_results=CONFIG.WEBSEARCH_MAX_RESULTS)
        web_results = websearch.get_response(query=query_text, context=query_text)['response']
        answer = (f'Here are some web search results (websearch data) that may help:\n\n {web_results}.\n There is no '
                  f'technical context for this query') if web_results else ''
        return answer


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
        result = await server.process_image_query(contents, query)

        # Log the similar images for debugging
        similar_images = result.get('similar_images', [])

        # Structure response to include similar images
        response = {
            "status": "success",
            "response": {
                "text_response": result.get('response', ''),
                "is_technical": result.get('is_technical', False),
                "confidence": result.get('confidence', 0),
                "images": similar_images,  # Changed from similar_images to images to match frontend expectation
                "document_references": result.get('document_references', [])
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/document")
async def upload_document(file: UploadFile, folder: str = Form("")):
    """
    Handles uploading of documents to a specified folder on the server.
    """

    def validate_and_update_path(file_path, prefix: str = "DOC"):
        """
        Helper function - Validates if the file name matches the format XXXXXX-filename.extension.
        If not, assigns a new unique ID based on the current date-time and returns the updated path.

        :param file_path: str - Original file path
        :param prefix: str - Filename Prefix
        :return: str - Updated file path if the original didn't match the format; otherwise, the original path
        """
        # Extract directory, file name, and extension
        directory, full_filename = os.path.split(file_path)
        filename, extension = os.path.splitext(full_filename)

        # Define the regex pattern for validation
        pattern = r"^[a-zA-Z0-9]{6}-.+\..+$"

        # Check if the file name matches the pattern
        if re.match(pattern, full_filename):
            return file_path  # Return the original path if it matches

        # Generate a new unique ID
        unique_id = prefix + datetime.now().strftime("%d%m%y%H%M")

        # Create the new file name
        new_filename = f"{unique_id}-{filename}{extension}"

        # Construct the updated path
        updated_path = os.path.join(directory, new_filename)

        return updated_path

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


def check_processing_status():
    """Check if all necessary files and data exist after processing"""
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
async def process_documents():
    """
    Process documents asynchronously while maintaining metadata persistence.
    Returns a status response indicating success or failure.
    """
    process = None
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
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1"
            }
        )

        try:
            stdout, stderr = process.communicate()
        except Exception as e:
            if process:
                process.kill()
                _, _ = process.communicate()
            raise e

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
            def get_entry_key(element_entry):
                if element_entry.get('type') == 'image':
                    return f"image_{element_entry.get('image', {}).get('id')}"
                elif element_entry.get('type') == 'text-chunk':
                    return f"chunk_{element_entry.get('path')}_{element_entry.get('chunk')}"
                else:
                    content_str = json.dumps(element_entry.get('content', {}), sort_keys=True)
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
    finally:
        # Ensure process is properly cleaned up
        if process:
            try:
                process.kill()
            except:
                pass


@app.get("/get/documents")
async def get_documents(path: str = ""):
    """Get list of documents and folders with metadata recursively"""
    try:
        # Get total count from processed_files
        processed_files_path = CONFIG.PROCESSED_FILES_PATH
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
