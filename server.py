import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import json
from dotenv import load_dotenv
from PIL import Image
from typing import Optional, Union, List, Dict, Tuple
import base64
from io import BytesIO
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import re
from pathlib import Path

from config import CONFIG
from utils.FAISS_utils import load_faiss_index, load_metadata, query_with_context
from utils.LLM_utils import CLIP_init, openai_post_request
from image_store import ImageStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG.LOG_PATH)
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
    images: List[Dict[str, str]] = []


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
            content = re.sub(r'(?m)((?:^|\n)(?:\d+\.|\•)[^\n]+)(?:\n(?!\d+\.|\•|$))', r'\1\n', content)
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
        self.index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
        self.metadata = load_metadata(CONFIG.METADATA_PATH)
        self.image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)
        self.similarity_threshold = 0.3  # Lowered threshold for better image matching
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
        """Get relevant contexts and images based on similarity scores"""
        if not results or not results[0]:
            return [], []

        relevant_contexts = []
        relevant_images = []

        # Keywords for better image matching
        keywords = set(query_text.lower().split())
        image_keywords = {'image', 'picture', 'diagram', 'photo', 'figure', 'illustration', 'logo', 'symbol', 'hazard'}
        is_image_query = bool(keywords & image_keywords)

        # Debug: Print all image entries in metadata
        image_entries = [m for m in self.metadata if m.get('type') == 'image']
        logging.info(f"Found {len(image_entries)} total image entries")
        logging.info("Sample of first 3 image entries:")
        for entry in image_entries[:3]:
            logging.info(f"Image entry: {json.dumps(entry, indent=2)}")

        # Debug: Print search results
        logging.info(f"Search results: {len(results[0])} items")
        for idx, result in enumerate(results[0][:3]):
            logging.info(
                f"Result {idx}: distance={result['distance']}, metadata={json.dumps(result['metadata'], indent=2)}")

        for result in results[0]:
            try:
                similarity_score = 1 / (1 + result['distance'])
                metadata = result['metadata']

                # Much lower threshold for image queries
                threshold = 0.2 if is_image_query else self.similarity_threshold

                if similarity_score >= threshold:
                    if metadata.get('type') == 'text-chunk':
                        content = metadata.get('content', '').strip()
                        if content and content not in relevant_contexts:
                            relevant_contexts.append(content)

                    elif metadata.get('type') == 'image':
                        # Try multiple ways to get image_id
                        image_id = None

                        # Debug: Print image metadata
                        logging.info(f"Processing image metadata: {json.dumps(metadata, indent=2)}")

                        # Check all possible locations for image_id
                        if 'image_id' in metadata:
                            image_id = metadata['image_id']
                        elif isinstance(metadata.get('content'), dict):
                            image_id = metadata['content'].get('image_id')
                        elif isinstance(metadata.get('content'), str) and 'images' in metadata.get('content', ''):
                            image_id = metadata.get('content').split('/')[-1].strip()

                        logging.info(f"Found image_id: {image_id}")

                        if image_id:
                            base64_image = self.image_store.get_base64_image(image_id)
                            if base64_image:
                                image_data = {
                                    'image': base64_image,
                                    'caption': metadata.get('caption', ''),
                                    'source': metadata.get('source_doc', ''),
                                    'similarity': similarity_score,
                                    'context': metadata.get('context', '')
                                }
                                relevant_images.append(image_data)
                                logging.info(f"Successfully added image {image_id}")
                            else:
                                logging.warning(f"Failed to get base64 image for ID: {image_id}")

            except Exception as e:
                logging.error(f"Error processing result: {e}", exc_info=True)
                continue

        if is_image_query:
            relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
            relevant_images = relevant_images[:5]  # Limit to top 5 most relevant images

        logging.info(f"Found {len(relevant_contexts)} text chunks and {len(relevant_images)} images")
        return relevant_contexts, relevant_images

    def get_image_data(self, image_id: str, metadata: Dict, similarity: float) -> Optional[Dict]:
        try:
            base64_image = self.image_store.get_base64_image(image_id)
            if not base64_image:
                return None

            image_data = {
                'image': base64_image,
                'caption': metadata.get('caption', ''),
                'source': metadata.get('source_doc', ''),
                'similarity': similarity,
                'context': metadata.get('context', '')
            }

            if isinstance(metadata.get('content'), dict):
                content = metadata['content']
                if content.get('context'):
                    image_data['context'] = content['context']
                if content.get('caption'):
                    image_data['caption'] = content['caption']

            return image_data
        except Exception as e:
            logging.error(f"Error getting image data: {e}")
            return None

    def prepare_prompt(self, query_text: str, contexts: List[str], query_type: QueryType, images: List[Dict]) -> str:
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
            "5. Keep responses concise and well-formatted"
        ]

        if query_type.is_technical:
            instructions.extend([
                "6. Focus on technical details and specifications",
                "7. Include step-by-step explanations if applicable"
            ])

        prompt = (
            f"Query: {query_text}\n\n"
            f"Available Documentation:\n{context_text}\n"
            f"{image_context}\n\n"
            f"Instructions:\n{chr(10).join(instructions)}\n\n"
            "Response:"
        )
        return prompt

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a technical documentation assistant specialized in providing "
                    "structured, accurate information. Format your responses with clear "
                    "sections using Markdown-style headers (##) and bullet points (•). "
                    "Reference specific documents and images when relevant."
                )
            },
            {"role": "user", "content": prompt}
        ]

    def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> QueryResponse:
        try:
            image_entries = [m for m in self.metadata if m.get('type') == 'image']
            logging.info(f"Total images in metadata: {len(image_entries)}")

            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                top_k=top_k * 2
            )

            contexts, images = self.get_relevant_contexts(results, query_text)

            if not contexts and not images:
                return QueryResponse(
                    text_response="No relevant information found.",
                    images=[]
                )

            query_type = self.determine_query_type(query_text)
            prompt = self.prepare_prompt(query_text, contexts, query_type, images)

            response = openai_post_request(
                messages=self.prepare_messages(prompt),
                model_name=CONFIG.GPT_MODEL,
                max_tokens=CONFIG.DETAIL_MAX_TOKENS,
                temperature=0.3 if query_type.is_technical else 0.7,
                api_key=self.openai_api_key
            )

            formatted_response = self.formatter.format_response(
                response['choices'][0]['message']['content'].strip()
            )

            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({"role": "assistant", "content": formatted_response})

            return QueryResponse(
                text_response=formatted_response,
                images=images
            )

        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

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


# Initialize FastAPI app
app = FastAPI(title="Atlantium RAG API")

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
    try:
        response = server.process_text_query(query)
        return response
    except Exception as e:
        logging.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/image")
async def image_query(image: UploadFile = File(...), query: Optional[str] = Form(None)):
    try:
        image_data = await image.read()
        response = await server.process_image_query(image_data, query)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error processing image query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_chat_history():
    history = server.get_chat_history()
    return {"history": history}


# Server startup event
@app.on_event("startup")
async def startup_event():
    logging.info("Starting RAG Query Server...")
    # Verify directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs(CONFIG.STORED_IMAGES_PATH, exist_ok=True)
    logging.info("Directory structure verified")


# Server shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down RAG Query Server...")


# Run server configuration
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=CONFIG.SERVER_PORT,
        log_level="info"
    )