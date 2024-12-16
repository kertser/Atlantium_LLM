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

import os
import sys
from pathlib import Path
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import json
from dotenv import load_dotenv
from PIL import Image
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import field
import base64
from io import BytesIO
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi import Body
from pydantic import BaseModel
from contextlib import asynccontextmanager
import shutil
import hmac
import hashlib
import subprocess
import faiss
from datetime import datetime
import re
from config import CONFIG
from utils.FAISS_utils import load_faiss_index, load_metadata, query_with_context
from utils.LLM_utils import CLIP_init, openai_post_request
from utils.image_store import ImageStore
from utils.image_utils import deduplicate_images, zero_shot_classification
from models.prompt_loader import PromptLoader
from models.prompts import PromptBuilder
from urllib.parse import unquote, quote
from utils.document_utils import (
    open_file_with_default_program,
    remove_document_from_rag,
    delete_folder_from_rag,
    rename_folder_in_rag,
    create_folder,
    validate_folder_name
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            CONFIG.LOG_PATH,
            maxBytes=2 * 1024 * 1024,  # 2 MB
            backupCount=3,  # Keep up to 3 old log files
            encoding='utf-8'
        )
    ]
)

def clean_path(path: str) -> str:
    """
    Sanitizes a file path by removing traversal sequences and replacing separators
    with the system-specific separator.

    Args:
        path (str): The path to be cleaned.

    Returns:
        str: The sanitized file path.
    """
    cleaned = path.replace('..', '').strip('/').strip('\\')
    return cleaned.replace('\\', os.path.sep).replace('/', os.path.sep)

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
        def clean_text(text: str) -> str:
            # Clean up excess whitespace while preserving structure
            text = re.sub(r'\s*\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()

        def format_lists(content: str) -> str:
            # Add <br> before valid numbered list items (number followed by dot and space)
            content = re.sub(r'([^\n])\s*(\d+\.\s+(?=[A-Za-z]))', r'\1<br>\2', content)

            # Add <br> before headers (## or ### or **)
            content = re.sub(r'([^\n])\s*(#{2, 3}\s+)', r'\1<br>\2', content)

            # Format bullet points with proper indentation
            content = re.sub(r'(?m)^[•\-]\s*', r'  • ', content)

            # Remove bulet points from bold text items with bulets
            content = re.sub(r'\*\*\s*•\s*', r'• ', content)

            # Format numbered lists with proper indentation, including bold numbered text
            content = re.sub(r'(?m)^(\d+\.\s+)(\*\*.*?\*\*)', r'    \1\2', content)

            # Ensure line breaks between list items, including bold list items
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

        self.image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)
        self.similarity_threshold = CONFIG.SIMILARITY_THRESHOLD
        self.formatter = EnhancedResponseFormatter()
        self.reset_chat()

        logging.info(
            f"Server initialized with {len([m for m in self.metadata if m.get('type') == 'image'])} images in metadata")

    def determine_query_type(self, query_text: str) -> QueryType:
        query_lower = query_text.lower()
        return QueryType(
            is_overview="overview" in query_lower or "what is" in query_lower,
            is_technical="technical" in query_lower or "how" in query_lower,
            is_summary="summary" in query_lower or "brief" in query_lower
        )

    def get_relevant_contexts(self, results: List[Dict], query_text: str) -> Tuple[List[str], List[Dict]]:
        """Get relevant contexts and images, filtering out non-technical images."""
        if not results or not results[0]:
            logging.info("No results found")
            return [], []

        logging.info(f"Processing query: {query_text}")
        logging.info(f"Found {len(results[0])} results")

        # Labels for zero-shot classification
        labels = [
            "a technical image",
            "a non-technical image",
        ]

        # Truncate query text to fit CLIP model's maximum length
        truncated_query = ' '.join(query_text.split()[:50])  # Approximate token limit

        try:
            # Get CLIP embedding for query
            query_input = self.processor(
                text=[truncated_query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's maximum sequence length
            )
            query_input = {k: v.to(self.device) for k, v in query_input.items()}
            query_embedding = self.model.get_text_features(**query_input)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

            relevant_contexts = []
            relevant_images = []

            for result in results[0]:
                metadata = result['metadata']
                logging.info(f"Metadata type: {metadata.get('type')}")

                if metadata.get('type') == 'text-chunk':
                    if result['distance'] < 1 / CONFIG.SIMILARITY_THRESHOLD:
                        relevant_contexts.append(metadata.get('content', '').strip())
                        logging.info("Added text context")

                elif metadata.get('type') == 'image':
                    logging.info(f"Image metadata: {json.dumps(metadata, indent=2)}")
                    image_id = metadata.get('image_id') or metadata.get('content', {}).get('image_id')
                    logging.info(f"Image ID: {image_id}")

                    if image_id:
                        try:
                            image, img_metadata = self.image_store.get_image(image_id)
                            if image:
                                logging.info(f"Successfully loaded image {image_id}")

                                # Perform zero-shot classification
                                predicted_label, confidence = zero_shot_classification(
                                    image=image,
                                    labels=labels,
                                    model=self.model,
                                    processor=self.processor,
                                    device=self.device
                                )

                                # Only process technical images
                                if predicted_label == "a technical image" and confidence > CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD:
                                    logging.info(
                                        f"Image {image_id} classified as technical with confidence {confidence}")

                                    image_input = self.processor(images=image, return_tensors="pt").to(self.device)
                                    image_embedding = self.model.get_image_features(**image_input)
                                    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

                                    similarity = (query_embedding @ image_embedding.T).item()
                                    logging.info(f"Similarity score for image {image_id}: {similarity}")

                                    if similarity > CONFIG.IMAGE_SIMILARITY_THRESHOLD:
                                        image_data = self.get_image_data(image_id, metadata, similarity)
                                        if image_data:
                                            image_data['technical_confidence'] = confidence
                                            relevant_images.append(image_data)
                                            logging.info(
                                                f"Added technical image {image_id} with similarity {similarity}")
                                else:
                                    logging.info(f"Skipped non-technical image {image_id} (confidence: {confidence})")
                            else:
                                logging.warning(f"Failed to load image {image_id}")
                        except Exception as e:
                            logging.error(f"Error processing image {image_id}: {e}")
                            continue

            relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
            logging.info(f"Found {len(relevant_contexts)} contexts and {len(relevant_images)} technical images")
            return relevant_contexts, relevant_images

        except Exception as e:
            logging.error(f"Error in get_relevant_contexts: {e}")
            return [], []

    def get_image_data(self, image_id: str, metadata: Dict, similarity: float) -> Optional[Dict]:
        """Get image data with improved logging"""
        try:
            logging.info(f"Retrieving image data for ID: {image_id}")
            base64_image = self.image_store.get_base64_image(image_id)
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

    def prepare_prompt(self, query_text: str, contexts: List[str], query_type: QueryType, images: List[Dict]) -> str:
        if not contexts and not images:
            # Use no-answer template if no relevant information
            return self.formatter.prompt_builder.build_no_answer_message(query_text)
        return self.formatter.prepare_prompt(query_text, contexts, query_type, images)

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        return self.formatter.prepare_messages(prompt)

    def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> QueryResponse:
        try:
            logging.info(f"Processing query: {query_text}")

            # Check if any documents are indexed
            if not self.metadata:
                return QueryResponse(
                    text_response="No documents have been indexed yet. Please add some documents to the system first.",
                    images=[]
                )

            # Get query results
            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                top_k=top_k * 2
            )

            # Handle empty results
            if not results:
                logging.info("No results found")
                text_response = self.formatter.prompt_builder.build_no_answer_message(query_text)[1]['content']
                return QueryResponse(text_response=text_response, images=[])

            # Get contexts and images
            contexts, images = self.get_relevant_contexts(results, query_text)

            # Handle conflict scenarios
            if len(contexts) > 1 and "conflicting" in query_text.lower():
                logging.info("Detected potential conflict in contexts")
                conflicting_docs = [{"doc": context} for context in contexts]
                conflict_response = self.formatter.prompt_builder.build_conflict_resolution_message(conflicting_docs)
                return QueryResponse(text_response=conflict_response[1]['content'], images=[])

            # Handle ambiguity scenarios
            if "ambiguous" in query_text.lower():
                logging.info("Detected ambiguous query")
                ambiguity_response = self.formatter.prompt_builder.build_ambiguity_message(query_text)
                return QueryResponse(text_response=ambiguity_response[1]['content'], images=[])

            # Deduplicate images
            if images:
                images = deduplicate_images(images)
                logging.info(f"After deduplication: {len(images)} unique images")

            # Prepare prompt and get GPT response
            query_type = self.determine_query_type(query_text)
            prompt = self.prepare_prompt(query_text, contexts, query_type, images)

            response = openai_post_request(
                messages=self.prepare_messages(prompt),
                model_name=CONFIG.GPT_MODEL,
                max_tokens=CONFIG.DETAIL_MAX_TOKENS,
                temperature=0.3 if query_type.is_technical else 0.7,
                api_key=self.openai_api_key
            )

            text_response = response['choices'][0]['message']['content'].strip()
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

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> str:
        try:
            # Open and process the image
            image = Image.open(BytesIO(image_data))
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert the processed image to a base64 string
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Prepare the image context using the PromptBuilder
            prompt_builder = PromptBuilder()
            image_context = prompt_builder.loader.format_template(
                'image_query',
                base64_image=f"data:image/jpeg;base64,{base64_image}",
                query_text=query_text or "What is shown in this image?"
            )

            # Build the final prompt
            full_prompt = prompt_builder.build_chat_prompt(
                query_text=query_text or "Analyze this image",
                contexts=[],
                images=[{"source": "uploaded", "context": "Uploaded Image"}],
                chat_history=self.chat_history,
                is_technical=False
            )

            # Make the API call with the generated prompt
            response = self.client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=prompt_builder.build_messages(full_prompt),
                max_tokens=CONFIG.VISION_MAX_TOKENS
            )

            answer = response.choices[0].message.content.strip()

            # Update chat history
            self.chat_history.append({
                "role": "user",
                "content": f"[Image Query] {query_text or 'Analyze image'}"
            })
            self.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            return answer

        except Exception as e:
            logging.error(f"Image processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

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

    Args:
        query (str): The user's query input.

    Returns:
        JSONResponse: Contains the generated text response and any relevant images.
    """
    try:
        # Process the query
        response = server.process_text_query(query)

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
        # Check file size (e.g., 5MB limit)
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 5MB"
            )

        # Verify file type
        try:
            img = Image.open(BytesIO(contents))
            if img.format.lower() not in ['jpeg', 'jpg', 'png', 'gif']:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Only JPEG, PNG and GIF are supported"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Process the image query with retries
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                response = await server.process_image_query(contents, query)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logging.warning(f"Retry {attempt + 1} after error: {str(e)}")
                await asyncio.sleep(1)

        if query:
            # Get the initial image context
            image_context = response

            # Use the image context along with the text query to get enhanced response
            enhanced_response = server.process_text_query(
                f"Context about the image: {image_context}\n\nUser query: {query}"
            )
            return {"response": enhanced_response.text_response}

        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing image query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


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
        # Sanitize and validate folder path
        clean_folder = folder.replace('..', '').strip('/').strip('\\')

        # Create full target directory path
        target_dir = CONFIG.RAW_DOCUMENTS_PATH
        if clean_folder:
            target_dir = target_dir / clean_folder
            # Ensure target directory exists and is within RAW_DOCUMENTS_PATH
            if not str(target_dir).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
                raise HTTPException(status_code=403, detail="Invalid folder path")
            target_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        sanitized_filename = os.path.basename(file.filename)
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

        return {
            "status": "success",
            "path": str(dest_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH))
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
async def process_documents():
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting document processing...")

        # Run RAG_processor.py with proper encoding environment variable
        process = subprocess.Popen(
            [sys.executable, 'RAG_processor.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        stdout, stderr = process.communicate()

        # Log the output properly
        if stdout:
            for line in stdout.splitlines():
                if 'ERROR' in line:
                    logger.error(line)
                else:
                    logger.info(line)

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

        # Wait a moment to ensure files are written
        await asyncio.sleep(1)

        # Verify the results
        success, message = check_processing_status()
        if not success:
            logger.error(f"Processing verification failed: {message}")
            raise HTTPException(status_code=500, detail=message)

        # Reload the server's index and metadata with proper encoding
        try:
            server.index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            server.metadata = load_metadata(CONFIG.METADATA_PATH)
        except Exception as e:
            logger.error(f"Error reloading index and metadata: {e}")
            raise HTTPException(status_code=500, detail="Failed to load processed data")

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
                "files": files
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


@app.post("/open/document")
async def open_document(path: str = Body(..., embed=True)):
    """Open a document with the default system program."""
    try:
        clean_file_path = clean_path(unquote(path))
        full_path = CONFIG.RAW_DOCUMENTS_PATH / clean_file_path

        # Security check
        if not str(full_path).startswith(str(CONFIG.RAW_DOCUMENTS_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        success, message = open_file_with_default_program(str(full_path))
        if not success:
            raise HTTPException(status_code=500, detail=message)

        return {"status": "success", "message": message}

    except Exception as e:
        logging.error(f"Error opening document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/document")
async def download_document(path: str):
    """Download a document."""
    try:
        # Clean and validate the path
        clean_file_path = clean_path(unquote(path))
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
        decoded_path = unquote(path)
        clean_file_path = clean_path(decoded_path)
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
        clean_parent_path = clean_path(unquote(parent_path))
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
        clean_folder_path = clean_path(unquote(path))
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
        # Decode the URL-encoded path
        decoded_path = unquote(path)
        clean_folder_path = clean_path(decoded_path)
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
async def rescan_documents():
    """Rescan for new documents and update RAG."""
    try:
        # Run RAG_processor.py in a subprocess
        process = subprocess.Popen(
            [sys.executable, 'RAG_processor.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        stdout, stderr = process.communicate()

        # Log the output
        if stdout:
            for line in stdout.splitlines():
                if 'ERROR' in line:
                    logging.error(line)
                else:
                    logging.info(line)

        if stderr:
            for line in stderr.splitlines():
                if 'ERROR' in line:
                    logging.error(f"Processing error: {line}")
                else:
                    logging.info(line)

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Rescan failed with code {process.returncode}"
            )

        # Wait for files to be written
        await asyncio.sleep(1)

        # Verify results
        success, message = check_processing_status()
        if not success:
            raise HTTPException(status_code=500, detail=message)

        return {"status": "success", "message": "Rescan completed successfully"}

    except Exception as e:
        logging.error(f"Error during rescan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_chat_history():
    """
    Retrieves the current chat history for the session.

    Returns:
        dict: Contains a list of chat history messages in chronological order.
    """
    history = server.get_chat_history()
    return {"history": history}


@app.get("/favicon.png")
async def favicon():
    return FileResponse("static/favicon.png", media_type="image/x-icon")


@app.post("/webhook")
async def github_webhook(request: Request):
    if WEBHOOK_SECRET:
        # Verify GitHub signature
        signature = request.headers.get('X-Hub-Signature-256')
        if not signature:
            raise HTTPException(status_code=403, detail="No signature provided")

        body = await request.body()
        hmac_gen = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256)
        expected_signature = f"sha256={hmac_gen.hexdigest()}"

        if not hmac.compare_digest(signature, expected_signature):
            raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        subprocess.run(["/usr/local/bin/update_rag.sh"], check=True)
        return {"status": "success"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run server configuration
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=CONFIG.SERVER_PORT,
        log_level="info"
    )
