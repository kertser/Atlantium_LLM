import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from dotenv import load_dotenv
from PIL import Image
from typing import Optional, Union
import base64
from io import BytesIO
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json

from config import CONFIG
from utils.FAISS_utils import load_faiss_index, load_metadata, query_with_context
from utils.LLM_utils import CLIP_init, openai_post_request

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

app = FastAPI(title="Atlantium RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # Initialize empty chat history
        self.reset_chat()

    def reset_chat(self):
        """Reset the chat history"""
        self.chat_history = []
        logging.info("Chat history has been reset")
        return {"status": "success", "message": "Chat history cleared"}

    def get_chat_history(self):
        """Return the chat history"""
        return self.chat_history

    def get_conversation_context(self, last_n=10) -> str:
        """Get the last N messages as context"""
        if not self.chat_history:
            return ""

        # Get last N messages
        recent_messages = self.chat_history[-last_n:] if len(self.chat_history) > last_n else self.chat_history

        # Format messages into a conversation string
        conversation = "\nPrevious conversation context:\n"
        for msg in recent_messages:
            conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"

        return conversation

    async def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> str:
        """Handle text-based queries"""
        try:
            # Get relevant context from FAISS
            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                top_k=top_k
            )

            if not results or not results[0]:
                context = "No relevant context found in the knowledge base."
            else:
                context = "\n\n".join([
                    res['metadata'].get('content', '')
                    for res in results[0]
                    if res['metadata'].get('content')
                ])

            # Add conversation history to context
            conversation_context = self.get_conversation_context()
            full_context = f"{context}\n\n{conversation_context}"

            prompt = (
                f"Technical Information Extraction Task:\n\n"
                f"Context:\n{full_context}\n\n"
                f"Query: {query_text}\n\n"
                f"Instructions:\n"
                "1. Analyze the provided context and previous conversation\n"
                "2. Extract relevant information for the query\n"
                "3. Provide a clear, concise response\n"
                "4. Reference previous conversation when relevant\n\n"
                "Response:"
            )

            messages = [
                {"role": "system", "content": "You are a precise technical information extraction assistant."},
                {"role": "user", "content": prompt}
            ]

            response = openai_post_request(
                messages=messages,
                model_name=CONFIG.GPT_MODEL,
                max_tokens=CONFIG.MAX_TOKENS,
                temperature=CONFIG.TEMPERATURE,
                api_key=self.openai_api_key
            )

            answer = response['choices'][0]['message']['content'].strip()

            # Update chat history
            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({"role": "assistant", "content": answer})

            return answer

        except Exception as e:
            logging.error(f"Error processing text query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_image_query(self, image: Union[Image.Image, bytes], query_text: Optional[str] = None) -> str:
        """Handle image-based queries"""
        try:
            # Convert bytes to PIL Image if necessary
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))

            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Add conversation history to context
            conversation_context = self.get_conversation_context()

            # Prepare the prompt with conversation context
            prompt = query_text or "Describe the contents and details of this image in a technical manner."
            if conversation_context:
                prompt = f"{prompt}\n\nConsider this conversation context when analyzing the image:{conversation_context}"

            # Image Analysis with OpenAI
            response = self.client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                        ],
                    }
                ],
                max_tokens=CONFIG.MAX_TOKENS
            )

            answer = response.choices[0].message.content.strip()

            # Update chat history
            self.chat_history.append(
                {"role": "user", "content": f"[Image Query] {query_text or 'Analyze this image'}"})
            self.chat_history.append({"role": "assistant", "content": answer})

            return answer

        except Exception as e:
            logging.error(f"Image processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize server
server = RAGQueryServer()

# API endpoints
@app.post("/chat/reset")
async def reset_chat():
    """Endpoint to reset the chat history"""
    return server.reset_chat()

@app.post("/query/text")
async def text_query(query: str = Form(...), top_k: int = CONFIG.DEFAULT_TOP_K):
    """Endpoint for text queries"""
    response = await server.process_text_query(query, top_k)
    return {"response": response}

@app.post("/query/image")
async def image_query(image: UploadFile = File(...), query: Optional[str] = Form(None)):
    """Endpoint for image queries"""
    image_data = await image.read()
    response = await server.process_image_query(image_data, query)
    return {"response": response}

@app.get("/chat/history")
async def get_chat_history():
    """Endpoint to retrieve chat history"""
    history = server.get_chat_history()
    return {"history": history}  # Make sure history is a list of message dictionaries