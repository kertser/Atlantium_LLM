# Set OpenMP environment variable before any other imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from dotenv import load_dotenv
from PIL import Image
from typing import Optional, Union
import base64
from io import BytesIO
from openai import OpenAI

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

    def process_query(self,
                     input_data: Union[str, Image.Image],
                     query_text: Optional[str] = None,
                     top_k: int = CONFIG.DEFAULT_TOP_K) -> str:
        """Process a query using the RAG system."""
        try:
            # Handle image input differently
            if isinstance(input_data, Image.Image):
                return self._process_image_query(input_data, query_text)
            else:
                return self._process_text_query(input_data, query_text, top_k)
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"

    def _process_text_query(self, input_text: str, query_text: Optional[str], top_k: int) -> str:
        """Handle text-based queries"""
        # Get relevant context from FAISS
        results = query_with_context(
            index=self.index,
            metadata=self.metadata,
            model=self.model,
            processor=self.processor,
            device=self.device,
            text_query=input_text,
            top_k=top_k
        )

        if not results or not results[0]:
            return "No relevant context found for the query."

        # Prepare context from results
        context = "\n\n".join([
            res['metadata'].get('content', '')
            for res in results[0]
            if res['metadata'].get('content')
        ])

        # Prepare prompt
        prompt = (
            f"Technical Information Extraction Task:\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query_text or input_text}\n\n"
            f"Instructions:\n"
            "1. Analyze the provided context\n"
            "2. Extract relevant information for the query\n"
            "3. Provide a clear, concise response\n\n"
            "Response:"
        )

        # Get response from OpenAI
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

        return response['choices'][0]['message']['content'].strip()

    def _process_image_query(self, image: Image.Image, query_text: Optional[str]) -> str:
        """Handle image-based queries"""
        try:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Image Analysis with OpenAI
            response = self.client.chat.completions.create(
                model=CONFIG.GPT_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query_text or "Describe the contents and details of this image in a technical manner."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                        ],
                    }
                ],
                max_tokens=CONFIG.MAX_TOKENS
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"Image processing error: {e}")
            return f"An error occurred while analyzing the image: {e}"

def main():
    try:
        # Initialize server
        server = RAGQueryServer()

        # Example text query
        print("\nTesting text query...")
        text_response = server.process_query(
            input_data="What are the voltage requirements?",
            top_k=CONFIG.DEFAULT_TOP_K
        )
        print("Text Query Response:", text_response)

        # Example image query
        print("\nTesting image query...")
        try:
            image = Image.open("image.png")
            image_response = server.process_query(
                input_data=image,
                query_text="What is shown in this image in context of Atlantium?",
                top_k=CONFIG.DEFAULT_TOP_K
            )
            print("Image Query Response:", image_response)
        except FileNotFoundError:
            print("Image file not found. Skipping image query test.")
        except Exception as e:
            print(f"Error processing image query: {e}")

    except Exception as e:
        print(f"Server initialization error: {e}")
        logging.error(f"Server initialization error: {e}", exc_info=True)

if __name__ == "__main__":
    main()