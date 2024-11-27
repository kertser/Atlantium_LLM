import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from dotenv import load_dotenv
from PIL import Image
from typing import Optional, Union, List, Dict
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
            # Remove multiple spaces and clean up newlines
            text = re.sub(r'\s+', ' ', text.strip())
            # Add proper line breaks after punctuation
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
                    # Handle nested points if they exist
                    if len(sub_points) > 1:
                        nested_points = [p.strip() for p in sub_points[1:] if p.strip()]
                        if nested_points:
                            formatted_points.extend(format_bullet_points(nested_points, indent_level + 1).split('\n'))
            return '\n'.join(formatted_points)

        # Split content into sections
        sections = re.split(r'(?m)^#{2,3}\s+', content)

        formatted_sections = []
        for section in sections:
            if not section.strip():
                continue

            # Extract title and content
            parts = section.split('\n', 1)
            if len(parts) > 1:
                title, content = parts
            else:
                continue

            # Format bullet points if present
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
            is_overview="about" in query_lower or "what is" in query_lower,
            is_technical="technical" in query_lower or "specification" in query_lower,
            is_summary="summary" in query_lower or "brief" in query_lower
        )

    def get_response_template(self, query_type: QueryType) -> str:
        """Get appropriate response template based on query type"""
        if query_type.is_overview:
            return (
                "{overview}\n\n"
                "{components}\n\n"
                "{features}\n\n"
                "{additional_info}"
            )
        elif query_type.is_technical:
            return (
                "{overview}\n\n"
                "{technical_specs}\n\n"
                "{installation}\n\n"
                "{maintenance}"
            )
        elif query_type.is_summary:
            return (
                "{overview}\n\n"
                "{key_points}\n"
            )
        return (
            "{overview}\n\n"
            "{details}\n"
        )

    def prepare_prompt(self, query_text: str, contexts: List[str], query_type: QueryType) -> str:
        """Prepare a structured prompt based on query type"""
        context_text = "\n\n".join(contexts) if contexts else "No relevant technical documentation found."
        conversation_context = self.get_conversation_context()

        # Base instructions
        instructions = [
            "1. Start with a clear, concise overview",
            "2. Use proper technical terminology",
            "3. Maintain accuracy and clarity",
            "4. Format the response with clear sections",
        ]

        # Add type-specific instructions
        if query_type.is_overview:
            instructions.extend([
                "5. Include comprehensive component descriptions",
                "6. List key features and capabilities",
            ])
        elif query_type.is_technical:
            instructions.extend([
                "5. Focus on technical specifications",
                "6. Include detailed operational parameters",
            ])
        elif query_type.is_summary:
            instructions.extend([
                "5. Keep information concise and focused",
                "6. Highlight only the most important points",
            ])

        prompt = (
            f"Technical Documentation Query\n\n"
            f"Query: {query_text}\n\n"
            f"Available Documentation Context:\n{context_text}\n\n"
            f"Previous Conversation:\n{conversation_context}\n\n"
            f"Response Format:\n"
            "- Use '## SECTION_NAME' for main sections\n"
            "- Use '### Subsection_Name' for subsections\n"
            "- Use bullet points (•) for lists\n"
            "- Use bold (**) for emphasis\n\n"
            f"Instructions:\n{chr(10).join(instructions)}\n\n"
            "Response:"
        )
        return prompt

    def post_process_response(self, response: str, query_type: QueryType) -> str:
        """Clean and format the response"""
        # Ensure consistent section headers
        response = re.sub(r'^(.*?):', lambda m: f"## {m.group(1).upper()}", response, flags=re.MULTILINE)

        # Format bullet points
        response = re.sub(r'^\s*[-•]\s*', '• ', response, flags=re.MULTILINE)

        # Format technical specifications
        response = re.sub(r'\b(\w+):\s', r'**\1:** ', response)

        # Add spacing
        response = re.sub(r'\n{3,}', '\n\n', response)

        # If it's a summary, ensure it's concise
        if query_type.is_summary:
            paragraphs = response.split('\n\n')
            if len(paragraphs) > 3:
                response = '\n\n'.join(paragraphs[:3])

        return response.strip()

    def get_relevant_contexts(self, results: List[Dict], query_text: str) -> List[str]:
        """Get relevant contexts based on similarity scores and adjacent chunks"""
        if not results or not results[0]:
            return []

        relevant_contexts = set()
        sorted_results = sorted(results[0], key=lambda x: x['distance'])

        for res in sorted_results:
            similarity_score = 1 / (1 + res['distance'])

            if similarity_score >= self.similarity_threshold:
                content = res['metadata'].get('content', '').strip()
                if content:
                    relevant_contexts.add(content)

                    # Find and add adjacent chunks
                    doc_chunks = [r['metadata'].get('content', '') for r in results[0]
                                  if r['metadata'].get('pdf') == res['metadata'].get('pdf')]

                    try:
                        current_idx = doc_chunks.index(content)
                        start_idx = max(0, current_idx - self.context_window)
                        end_idx = min(len(doc_chunks), current_idx + self.context_window + 1)

                        for chunk in doc_chunks[start_idx:end_idx]:
                            if chunk.strip():
                                relevant_contexts.add(chunk)
                    except ValueError:
                        continue

        return list(relevant_contexts)

    def prepare_messages(self, query_text: str, query_type: QueryType) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        # Get contexts and prepare prompt
        results = query_with_context(
            index=self.index,
            metadata=self.metadata,
            model=self.model,
            processor=self.processor,
            device=self.device,
            text_query=query_text,
            top_k=CONFIG.DEFAULT_TOP_K * 2
        )

        relevant_contexts = self.get_relevant_contexts(results, query_text)
        prompt = self.prepare_prompt(query_text, relevant_contexts, query_type)

        return [
            {
                "role": "system",
                "content": (
                    "You are a technical documentation assistant specialized in providing "
                    "structured, accurate information. Format your responses with clear "
                    "sections using Markdown-style headers (##) and bullet points (•). "
                    "Focus on clarity and technical accuracy."
                )
            },
            {"role": "user", "content": prompt}
        ]

    async def process_text_query(self, query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> str:
        try:
            # Determine query type
            query_type = self.determine_query_type(query_text)

            # Get relevant context from FAISS with increased results
            results = query_with_context(
                index=self.index,
                metadata=self.metadata,
                model=self.model,
                processor=self.processor,
                device=self.device,
                text_query=query_text,
                top_k=top_k * 2
            )

            # Get relevant contexts
            relevant_contexts = self.get_relevant_contexts(results, query_text)

            # Prepare the prompt
            prompt = self.prepare_prompt(query_text, relevant_contexts, query_type)

            # Set temperature based on query type
            temperature = 0.3 if query_type.is_technical else 0.7 if query_type.is_overview else 0.5

            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": (
                    "You are a technical documentation assistant specialized in providing "
                    "structured, accurate information. Format your responses with clear "
                    "sections using Markdown-style headers (##) and bullet points (•). "
                    "Focus on clarity and technical accuracy."
                )},
                {"role": "user", "content": prompt}
            ]

            # Get response from OpenAI with increased token limit
            response = openai_post_request(
                messages=messages,
                model_name=CONFIG.GPT_MODEL,
                max_tokens=1500,  # Increased token limit
                temperature=temperature,
                api_key=self.openai_api_key
            )

            # Format the response using the EnhancedResponseFormatter
            formatter = EnhancedResponseFormatter()
            formatted_response = formatter.format_response(
                response['choices'][0]['message']['content'].strip()
            )

            # Update chat history
            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({"role": "assistant", "content": formatted_response})

            return formatted_response

        except Exception as e:
            logging.error(f"Error processing text query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_image_query(self, image_data: bytes, query_text: Optional[str] = None) -> str:
        try:
            # Open and convert image
            image = Image.open(BytesIO(image_data))

            # Convert RGBA to RGB if needed
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Save to buffer
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query_text or "What is shown in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                        }
                    }
                ]
            }]

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=500
                )
            except Exception as api_error:
                logging.error(f"OpenAI API error: {str(api_error)}")
                raise HTTPException(status_code=500, detail="Error processing image with OpenAI API")

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

    def get_conversation_context(self, last_n=5) -> str:
        """Get the last N messages as context"""
        if not self.chat_history:
            return ""

        recent_messages = self.chat_history[-last_n:] if len(self.chat_history) > last_n else self.chat_history
        conversation = "\nPrevious conversation context:\n"
        for msg in recent_messages:
            conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return conversation




# Initialize server
server = RAGQueryServer()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

# API endpoints
@app.post("/chat/reset")
async def reset_chat():
    """Endpoint to reset the chat history"""
    return server.reset_chat()


@app.post("/query/text")
async def text_query(query: str = Form(...)):
    try:
        response = await server.process_text_query(query)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    return {"history": history}

# Add UVICORN configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)