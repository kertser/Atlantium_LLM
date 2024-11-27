import os
import logging
from dotenv import load_dotenv
from PIL import Image
import base64
from openai import OpenAI
import re
from utils import LLM_utils, RAG_utils, FAISS_utils

# %% Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)

# %% Load environment variables
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # numpy issue patch

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not all([OPENAI_API_KEY]):
    raise ValueError("One or more environment variables are missing. Please check your .env file.")

# %% Initialize CLIP embedding model
clip_model, clip_processor, device = LLM_utils.CLIP_init()

# %% Load index and metadata from vectorstore
index = FAISS_utils.load_faiss_index("faiss_index.bin")
metadata = FAISS_utils.load_metadata("faiss_metadata.json")


# %% Helper Functions
def encode_image(image_path):
    """Convert an image file to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def relevance_filter(context_list, query_text):
    """
    Filter context based on its relevance to the query text.
    :param context_list: List of context strings retrieved from FAISS.
    :param query_text: The query text for relevance comparison.
    :return: List of relevant context strings.
    """
    if not query_text:
        return context_list

    query_words = set(query_text.lower().split())
    relevant_context = []

    for context in context_list:
        if not context:
            continue
        try:
            words_in_context = set(context.lower().split())
            relevance_score = len(query_words.intersection(words_in_context))
            if relevance_score > 1:  # Adjust threshold for relevance
                relevant_context.append(context)
        except AttributeError as e:
            logging.warning(f"Skipped context due to error: {e}")

    return relevant_context


# %% Updated `process_input` Function
def process_input(input_data, query_text=None, index=None, metadata=None, model=None, processor=None, device="cpu",
                  top_k=10, verbose=False):
    """
    Process input data (text or image) to generate a response with enhanced context retrieval and response generation.

    Args:
        input_data (str/Image): Input text or image to process
        query_text (str, optional): Specific query or question about the input
        index (FAISS index, optional): Vectorstore index for semantic search
        metadata (dict, optional): Metadata associated with the index
        model (object, optional): Embedding model for semantic search
        processor (object, optional): Processor for the embedding model
        device (str, optional): Computation device (cpu/cuda)
        top_k (int, optional): Number of top context results to retrieve
        verbose (bool, optional): Enable detailed logging

    Returns:
        str: Generated response based on input and context
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Verbose logging
    def log_verbose(message):
        if verbose:
            logging.info(message)

    # Detect input type
    if isinstance(input_data, str):
        # Check if string is an image path
        if input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_type = 'image'
            input_data = Image.open(input_data)
        else:
            input_type = 'text'
    elif isinstance(input_data, Image.Image):
        input_type = 'image'
    else:
        raise ValueError("Unsupported input type. Expected a string or a PIL Image object.")

    # Text Input Processing
    if input_type == "text":
        log_verbose(f"Processing text input: {input_data}")

        # Semantic Search
        try:
            results = FAISS_utils.query_with_context(
                index=index,
                metadata=metadata,
                model=model,
                processor=processor,
                device=device,
                text_query=input_data,
                top_k=top_k
            )
            log_verbose(f"Retrieved {len(results[0])} context results")
        except Exception as e:
            logging.error(f"Context retrieval error: {e}")
            return f"Error retrieving context: {e}"

        # Extract and prepare context
        context_list = [
            res['metadata'].get('content', '')
            for res in results[0]
            if res['metadata'].get('content')
        ]

        # If no context found
        if not context_list:
            log_verbose("No context found for the query")
            return "No relevant technical context could be retrieved for this query."

        # Join contexts
        context = "\n\n".join(context_list)
        log_verbose(f"Total context length: {len(context)} characters")

        # Determine query text
        if not query_text:
            query_text = input_data

        # Enhanced Prompt for Technical Extraction
        prompt = (
            f"Technical Information Extraction Task:\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query_text}\n\n"
            f"Instructions:\n"
            "1. Carefully analyze the provided context\n"
            "2. Extract precise, factual information directly related to the query\n"
            "3. If the exact information is not available, state what is missing\n"
            "4. Prioritize technical accuracy over verbosity\n\n"
            "Extracted Technical Response:"
        )

        # Prepare messages for OpenAI
        messages = [
            {"role": "system",
             "content": "You are a precise technical information extraction assistant specializing in detailed, accurate responses."},
            {"role": "user", "content": prompt}
        ]

        # API Call with Enhanced Parameters
        try:
            response = LLM_utils.openai_post_request(
                messages=messages,
                model_name="gpt-4-turbo",  # Consider using the most capable model
                max_tokens=300,  # Increased for comprehensive responses
                temperature=0.1,  # Slight creativity while maintaining precision
                api_key=OPENAI_API_KEY
            )

            # Extract and clean response
            extracted_response = response['choices'][0]['message']['content'].strip()

            log_verbose(f"Generated response length: {len(extracted_response)}")
            return extracted_response if extracted_response else "No specific information could be extracted."

        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return f"Error generating response: {e}"

    # Image Input Processing (Remains similar to previous implementation)
    elif input_type == "image":
        if not query_text:
            query_text = "Describe the contents and details of this image in a technical manner."

        try:
            # Convert PIL Image to base64
            buffered = Image.io.BytesIO()
            input_data.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Image Analysis with OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                        ],
                    }
                ],
                max_tokens=300
            )

            # Extract and return response
            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"Image processing error: {e}")
            return f"An error occurred while analyzing the image: {e}"


# %% Example Queries
def main():
    # Logging configuration (if not already set up)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    # Text query example
    response = process_input(
        input_data="What are the voltage pins for RZ163?",
        index=index,
        metadata=metadata,
        model=clip_model,
        processor=clip_processor,
        device=device
    )
    print("Response:", response)


    # Image query example using a file path

    image_path = "image.png"
    response = process_input(
        input_data=image_path,
        query_text="What is shown in this image in context of Atlantium user interface",  # Optional specific question about the image
        index=index,
        metadata=metadata,
        model=clip_model,
        processor=clip_processor,
        device=device
    )
    print("Response:", response)


if __name__ == "__main__":
    main()