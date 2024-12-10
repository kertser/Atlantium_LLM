import os
import sys
from pathlib import Path
import asyncio
import logging
import json
from dotenv import load_dotenv
from PIL import Image
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import field
import base64
from io import BytesIO
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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
import imagehash
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG.LOG_PATH, encoding='utf-8')
    ]
)


# Data models
class ChatMessage(BaseModel):
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
    @staticmethod
    def format_response(content: str) -> str:
        def clean_text(text: str) -> str:
            # Clean up excess whitespace while preserving structure
            text = re.sub(r'\s*\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()

        def format_lists(content: str) -> str:
            # Format numbered lists and bullet points
            content = re.sub(r'(?m)^(\d+\.)\s*', r'\1 ', content)
            content = re.sub(r'(?m)^[•\-]\s*', '• ', content)
            # Ensure line breaks between list items
            content = re.sub(r'(?m)((?:^|\n)(?:\d+\.|•)[^\n]+)\n(?!\d+\.|•|$)', r'\1\n', content)
            return content

        def format_section(title: str, content: str) -> str:
            # Format section with consistent spacing
            formatted_content = clean_text(content)
            formatted_content = format_lists(formatted_content)
            return f"# {title}\n\n{formatted_content}"

        # Process the content
        sections = []
        current_title = "Overview"
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
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.openai_api_key)
        self.model, self.processor, self.device = CLIP_init(CONFIG.CLIP_MODEL_NAME)

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
        if not results or not results[0]:
            logging.info("No results found")
            return [], []

        logging.info(f"Processing query: {query_text}")
        logging.info(f"Found {len(results[0])} results")

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
                                image_input = self.processor(images=image, return_tensors="pt").to(self.device)
                                image_embedding = self.model.get_image_features(**image_input)
                                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

                                similarity = (query_embedding @ image_embedding.T).item()
                                logging.info(f"Similarity score for image {image_id}: {similarity}")

                                if similarity > CONFIG.IMAGE_SIMILARITY_THRESHOLD:
                                    image_data = self.get_image_data(image_id, metadata, similarity)
                                    if image_data:
                                        relevant_images.append(image_data)
                                        logging.info(f"Added image {image_id} with similarity {similarity}")
                            else:
                                logging.warning(f"Failed to load image {image_id}")
                        except Exception as e:
                            logging.error(f"Error processing image {image_id}: {e}")
                            continue

            relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
            logging.info(f"Found {len(relevant_contexts)} contexts and {len(relevant_images)} images")
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
        """Prepare prompt with deduplicated images."""
        # Get relevant chat history (last few exchanges)
        chat_context = ""
        if self.chat_history:
            last_exchanges = self.chat_history[-(2 * CONFIG.MAX_CHAT_HISTORY):]  # Get last Q&A pairs
            chat_context = "\nRecent Chat History:\n" + "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in last_exchanges
            ])

        context_text = "\n\n".join(contexts) if contexts else "No relevant technical documentation found."
        image_context = ""

        if images:
            image_descriptions = []
            for img in images:
                desc = f"- Image from {img['source']}"
                if img.get('caption'):
                    desc += f": {img['caption']}"
                if img.get('context'):
                    desc += f" (Context: {img['context']})"
                image_descriptions.append(desc)
            image_context = "\n\nRelevant Images:\n" + "\n".join(image_descriptions)

        instructions = [
            "1. Provide a clear and structured response",
            "2. Use technical terminology appropriately",
            "3. Reference specific documents and images when relevant",
            "4. Use section headers for organization",
            "5. Keep responses concise and well-formatted",
            "6. Consider previous chat context when relevant"
        ]

        if query_type.is_technical:
            instructions.extend([
                "7. Focus on technical details and specifications",
                "8. Include step-by-step explanations if applicable"
            ])

        prompt = (
            f"Query: {query_text}\n\n"
            f"Available Documentation:\n{context_text}\n"
            f"{image_context}\n"
            f"{chat_context}\n\n"
            f"Instructions:\n{chr(10).join(instructions)}\n\n"
            "Response:"
        )
        return prompt

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are Atlantium Technologies technical documentation assistant specialized in providing "
                    "structured, accurate information. Format your responses with clear "
                    "sections using Markdown-style headers (##) and bullet points (•). "
                    "Reference specific documents and images when relevant."
                )
            },
            {"role": "user", "content": prompt}
        ]

    def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> QueryResponse:
        """Process a text query and return response with relevant images"""
        try:
            logging.info(f"Processing query: {query_text}")

            # Check if any documents are indexed
            if not self.metadata:
                return QueryResponse(
                    text_response="No documents have been indexed yet. Please add some documents to the system first.",
                    images=[]
                )

            # Get query results
            try:
                results = query_with_context(
                    index=self.index,
                    metadata=self.metadata,
                    model=self.model,
                    processor=self.processor,
                    device=self.device,
                    text_query=query_text,
                    top_k=top_k * 2
                )
            except Exception as e:
                logging.error(f"Error in query_with_context: {e}")
                results = []

            # Handle empty results
            if not results:
                text_response = "No relevant information found. Try adding more documents or rephrasing your question."
                return QueryResponse(
                    text_response=text_response,
                    images=[]
                )

            # Get contexts and images
            contexts, images = self.get_relevant_contexts(results, query_text)

            # Deduplicate images right after getting them
            if images:
                images = deduplicate_images(images)
                logging.info(f"After deduplication: {len(images)} unique images")

            # Determine appropriate response based on available information
            if not contexts and images:
                text_response = "Here are the relevant images I found:"
            elif not contexts and not images:
                text_response = "I couldn't find any specific information about that. Could you please rephrase your question?"
            else:
                # Prepare prompt and get GPT response
                query_type = self.determine_query_type(query_text)
                prompt = self.prepare_prompt(query_text, contexts, query_type, images)

                try:
                    response = openai_post_request(
                        messages=self.prepare_messages(prompt),
                        model_name=CONFIG.GPT_MODEL,
                        max_tokens=CONFIG.DETAIL_MAX_TOKENS,
                        temperature=0.3 if query_type.is_technical else 0.7,
                        api_key=self.openai_api_key
                    )
                    text_response = response['choices'][0]['message']['content'].strip()
                    text_response = self.formatter.format_response(text_response)
                except Exception as e:
                    logging.error(f"Error getting GPT response: {e}")
                    text_response = "I found some relevant information but encountered an error processing it. Please try again."

            # Update chat history
            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({
                "role": "assistant",
                "content": text_response
            })

            logging.info(f"Returning response with {len(images)} deduplicated images")
            return QueryResponse(
                text_response=text_response,
                images=images
            )

        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return QueryResponse(
                text_response="An error occurred while processing your query. Please try again.",
                images=[]
            )

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> str:
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

            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text or "What is shown in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": CONFIG.VISION_QUALITY
                        }
                    }
                ]
            }]

            response = self.client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=messages,
                max_tokens=CONFIG.VISION_MAX_TOKENS
            )

            answer = response.choices[0].message.content.strip()

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


def calculate_image_hash(image_data: str) -> str:
    """Calculate perceptual hash from base64 image data"""
    try:
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Calculate multiple hash types for better accuracy
        avg_hash = str(imagehash.average_hash(image))
        dhash = str(imagehash.dhash(image))
        phash = str(imagehash.phash(image))

        return f"{avg_hash}_{dhash}_{phash}"
    except Exception as e:
        logging.error(f"Error calculating image hash: {e}")
        return None


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
app.mount("/static", StaticFiles(directory="static"), name="static")


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/chat/reset")
async def reset_chat():
    return server.reset_chat()


@app.post("/query/text")
async def text_query(query: str = Form(...)):
    """Handle text queries"""
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
async def upload_document(file: UploadFile):
    """Handle document upload"""
    try:
        # Ensure the directory exists
        CONFIG.RAW_DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        sanitized_filename = os.path.basename(file.filename)
        if not sanitized_filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Validate file extension
        if not any(sanitized_filename.lower().endswith(ext)
                   for ext in CONFIG.SUPPORTED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Create full path
        dest_path = CONFIG.RAW_DOCUMENTS_PATH / sanitized_filename

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

        return {"status": "success", "path": str(dest_path)}

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
        processed_files.update([str(path) for path in doc_paths])

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


def normalize_and_hash_image(image_data: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[str, Tuple[int, int]]:
    """
    Normalize image size and calculate perceptual hash.

    Args:
        image_data: Base64 encoded image data
        target_size: Size to normalize to before hashing

    Returns:
        Tuple of (hash_string, original_size)
    """
    try:
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        original_size = image.size

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize for consistent hashing
        normalized = image.resize(target_size, Image.Resampling.LANCZOS)

        # Calculate multiple hash types
        avg_hash = imagehash.average_hash(normalized)
        dhash = imagehash.dhash(normalized)
        phash = imagehash.phash(normalized)

        # Convert hashes to binary strings for more precise comparison
        hash_string = f"{avg_hash}_{dhash}_{phash}"

        return hash_string, original_size
    except Exception as e:
        logging.error(f"Error calculating image hash: {e}")
        return None, None


def are_images_similar(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """
    Compare image hashes to determine similarity.
    Lower threshold means stricter matching.
    """
    try:
        # Split combined hashes
        avg1, dhash1, phash1 = hash1.split('_')
        avg2, dhash2, phash2 = hash2.split('_')

        # Convert string hashes back to imagehash objects
        avg_diff = imagehash.hex_to_hash(avg1) - imagehash.hex_to_hash(avg2)
        dhash_diff = imagehash.hex_to_hash(dhash1) - imagehash.hex_to_hash(dhash2)
        phash_diff = imagehash.hex_to_hash(phash1) - imagehash.hex_to_hash(phash2)

        # Images are similar if any two hash types indicate similarity
        similarities = [diff <= threshold for diff in (avg_diff, dhash_diff, phash_diff)]
        return sum(similarities) >= 2
    except Exception as e:
        logging.error(f"Error comparing image hashes: {e}")
        return False


def deduplicate_images(images: list, max_images: int = 8) -> list:
    """
    Deduplicate images using perceptual hashing with size normalization.

    Args:
        images: List of image dictionaries
        max_images: Maximum number of images to return

    Returns:
        List of deduplicated images, limited to max_images
    """
    # Dictionary to track unique images by their perceptual hash
    unique_images = {}
    hash_groups = {}  # Track groups of similar images

    for img in images:
        image_id = img.get('image_id')
        if not image_id or not img.get('image'):
            continue

        # Calculate normalized hash and get original size
        image_hash, original_size = normalize_and_hash_image(img['image'])
        if not image_hash:
            continue

        # Check if this image is similar to any existing ones
        found_match = False
        for existing_hash in list(unique_images.keys()):  # Create list copy to avoid runtime modification issues
            if are_images_similar(image_hash, existing_hash):
                found_match = True
                # Use existing hash as the group key
                group_hash = existing_hash
                if img.get('similarity', 0) > unique_images[group_hash].get('similarity', 0):
                    # Keep metadata from old image
                    old_metadata = unique_images[group_hash]
                    unique_images[group_hash] = img
                    # Merge metadata
                    img = merge_image_metadata(img, old_metadata)

                # Track image ID in hash group
                if group_hash not in hash_groups:
                    hash_groups[group_hash] = set()
                hash_groups[group_hash].add(image_id)

                logging.info(f"Found similar images. ID: {image_id}, Size: {original_size}")
                break

        if not found_match:
            unique_images[image_hash] = img
            hash_groups[image_hash] = {image_id}
            logging.info(f"New unique image. ID: {image_id}, Size: {original_size}")

    # Log duplicate groups
    for hash_val, ids in hash_groups.items():
        if len(ids) > 1:
            logging.info(f"Duplicate image group with {len(ids)} variants: {ids}")

    # Convert to list and sort by similarity
    deduplicated = list(unique_images.values())
    deduplicated.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    logging.info(f"Deduplicated {len(images)} images to {len(deduplicated)} unique images")
    return deduplicated[:max_images]


def merge_image_metadata(primary_img: dict, secondary_img: dict) -> dict:
    """Merge metadata from two image records."""
    merged = primary_img.copy()

    # Merge captions
    if secondary_img.get('caption'):
        captions = set(primary_img.get('caption', '').split(' | '))
        captions.add(secondary_img['caption'])
        merged['caption'] = ' | '.join(filter(None, captions))

    # Merge contexts
    if secondary_img.get('context'):
        contexts = set(primary_img.get('context', '').split(' | '))
        contexts.add(secondary_img['context'])
        merged['context'] = ' | '.join(filter(None, contexts))

    # Merge sources
    if secondary_img.get('source'):
        sources = set(primary_img.get('source', '').split(' | '))
        sources.add(secondary_img['source'])
        merged['source'] = ' | '.join(filter(None, sources))

    return merged

@app.post("/process/documents")
async def process_documents():
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting document processing...")

        # Run rag_system.py with proper encoding environment variable
        process = subprocess.Popen(
            [sys.executable, 'rag_system.py'],
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
async def get_documents():
    """Get list of documents with metadata"""
    logger = logging.getLogger(__name__)
    try:
        documents = []
        logger.info(f"Scanning directory: {CONFIG.RAW_DOCUMENTS_PATH}")

        # Ensure the directory exists
        CONFIG.RAW_DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

        # Get all files with supported extensions
        for ext in CONFIG.SUPPORTED_EXTENSIONS:
            for file_path in CONFIG.RAW_DOCUMENTS_PATH.glob(f"*{ext}"):
                try:
                    stat = file_path.stat()
                    documents.append({
                        "name": file_path.name,
                        "type": file_path.suffix[1:].upper(),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

        logger.info(f"Found {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
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

@app.get("/chat/history")
async def get_chat_history():
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
