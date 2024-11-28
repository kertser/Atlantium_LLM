import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
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


# Define data models
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


app = FastAPI(title="Atlantium RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EnhancedResponseFormatter:
    @staticmethod
    def format_response(content: str) -> str:
        """Format the response with proper structure and styling"""

        def clean_section(text: str) -> str:
            """Clean and format a section of text"""
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'([.!?])\s+', r'\1\n', text)
            return text

        def format_section(title: str, content: str, level: int = 1) -> str:
            """Format a section with title and content"""
            header = '#' * level
            formatted_content = clean_section(content)
            return f"\n{header} {title}\n\n{formatted_content}\n"

        def format_bullet_points(points: List[str], indent_level: int = 0) -> str:
            """Format a list of bullet points"""
            indent = '  ' * indent_level
            formatted_points = []
            for point in points:
                if point.strip():
                    sub_points = point.split('\n')
                    main_point = sub_points[0].strip()
                    formatted_points.append(f"{indent}• {main_point}")
                    if len(sub_points) > 1:
                        nested_points = [p.strip() for p in sub_points[1:] if p.strip()]
                        if nested_points:
                            formatted_points.extend(format_bullet_points(nested_points, indent_level + 1).split('\n'))
            return '\n'.join(formatted_points)

        sections = re.split(r'(?m)^#{2,3}\s+', content)
        formatted_sections = []

        for section in sections:
            if not section.strip():
                continue
            parts = section.split('\n', 1)
            if len(parts) > 1:
                title, content = parts
                if '•' in content or '-' in content:
                    points = re.split(r'(?m)^[•-]\s*', content)
                    points = [p for p in points if p.strip()]
                    formatted_content = format_bullet_points(points)
                else:
                    formatted_content = clean_section(content)
                formatted_sections.append(format_section(title.strip(), formatted_content))

        return '\n'.join(formatted_sections)


class RAGQueryServer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)

        # Initialize CLIP model
        self.model, self.processor, self.device = CLIP_init(CONFIG.CLIP_MODEL_NAME)
        logging.info(f"CLIP model initialized on {self.device}")

        # Load FAISS index and metadata
        self.index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
        self.metadata = load_metadata(CONFIG.METADATA_PATH)
        logging.info("FAISS index and metadata loaded successfully")

        # Initialize image store
        self.image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)
        logging.info("Image store initialized")

        # Initialize empty chat history
        self.reset_chat()

        # Configure similarity threshold
        self.similarity_threshold = 0.7
        self.context_window = 2

        # Initialize formatter
        self.formatter = EnhancedResponseFormatter()

    def determine_query_type(self, query_text: str) -> QueryType:
        """Determine the type of query to adjust response format"""
        query_lower = query_text.lower()
        return QueryType(
            is_overview="overview" in query_lower or "what is" in query_lower,
            is_technical="technical" in query_lower or "how" in query_lower,
            is_summary="summary" in query_lower or "brief" in query_lower
        )

    def prepare_prompt(self, query_text: str, contexts: List[str], query_type: QueryType, images: List[Dict]) -> str:
        """Prepare a structured prompt based on query type and available content"""
        context_text = "\n\n".join(contexts) if contexts else "No relevant technical documentation found."
        image_context = ""
        if images:
            image_context = "\n\nRelevant Images:\n" + "\n".join(
                f"- Image from {img['source']} with similarity {img['similarity']:.2f}" +
                (f": {img['caption']}" if img['caption'] else "")
                for img in images
            )

        instructions = [
            "1. Provide a clear and structured response",
            "2. Use technical terminology appropriately",
            "3. Reference specific documents and images when relevant",
            "4. Format with clear section headers"
        ]

        if query_type.is_technical:
            instructions.extend([
                "5. Focus on technical details and specifications",
                "6. Include step-by-step explanations if applicable"
            ])
        elif query_type.is_overview:
            instructions.extend([
                "5. Provide a high-level overview",
                "6. Highlight key features and capabilities"
            ])

        prompt = (
            f"Query: {query_text}\n\n"
            f"Available Documentation:\n{context_text}\n"
            f"{image_context}\n\n"
            f"Instructions:\n{chr(10).join(instructions)}\n\n"
            "Response:"
        )
        return prompt

    def get_relevant_contexts(self, results: List[Dict], query_text: str) -> Tuple[List[str], List[Dict]]:
        """Get relevant contexts and images based on similarity scores"""
        if not results or not results[0]:
            return [], []

        relevant_contexts = set()
        relevant_images = []
        sorted_results = sorted(results[0], key=lambda x: x['distance'])

        for res in sorted_results:
            similarity_score = 1 / (1 + res['distance'])

            if similarity_score >= self.similarity_threshold:
                metadata = res['metadata']

                if metadata.get('type') == 'text-chunk':
                    content = metadata.get('content', '').strip()
                    if content:
                        relevant_contexts.add(content)

                elif metadata.get('type') == 'image':
                    image_id = metadata.get('image_id')
                    if image_id:
                        base64_image = self.image_store.get_base64_image(image_id)
                        if base64_image:
                            relevant_images.append({
                                'image': base64_image,
                                'caption': metadata.get('caption', ''),
                                'source': metadata.get('source_doc', ''),
                                'similarity': similarity_score
                            })

        return list(relevant_contexts), relevant_images

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
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

    async def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> QueryResponse:
        """Process a text query and return response with relevant images"""
        try:
            query_type = self.determine_query_type(query_text)
            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                top_k=top_k * 2
            )

            relevant_contexts, relevant_images = self.get_relevant_contexts(results, query_text)
            prompt = self.prepare_prompt(query_text, relevant_contexts, query_type, relevant_images)

            temperature = 0.3 if query_type.is_technical else 0.7 if query_type.is_overview else 0.5

            response = openai_post_request(
                messages=self.prepare_messages(prompt),
                model_name=CONFIG.GPT_MODEL,
                max_tokens=CONFIG.DETAIL_MAX_TOKENS,
                temperature=temperature,
                api_key=self.openai_api_key
            )

            formatted_response = self.formatter.format_response(
                response['choices'][0]['message']['content'].strip()
            )

            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({"role": "assistant", "content": formatted_response})

            return QueryResponse(
                text_response=formatted_response,
                images=relevant_images
            )

        except Exception as e:
            logging.error(f"Error processing text query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> str:
        """Process an image query and return response"""
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
        """Reset the chat history"""
        self.chat_history = []
        logging.info("Chat history has been reset")
        return {"status": "success", "message": "Chat history cleared"}

    def get_chat_history(self):
        """Return the chat history"""
        return self.chat_history


# Initialize server
server = RAGQueryServer()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint serving the main HTML interface"""
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/chat/reset")
async def reset_chat():
    """Reset the chat history"""
    return server.reset_chat()


@app.post("/query/text")
async def text_query(query: str = Form(...)):
    """Handle text queries"""
    try:
        response = await server.process_text_query(query)
        return response
    except Exception as e:
        logging.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/image")
async def image_query(image: UploadFile = File(...), query: Optional[str] = Form(None)):
    """Handle image queries"""
    try:
        image_data = await image.read()
        response = await server.process_image_query(image_data, query)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error processing image query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_chat_history():
    """Retrieve chat history"""
    history = server.get_chat_history()
    return {"history": history}


# Run server configuration
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=CONFIG.SERVER_PORT)