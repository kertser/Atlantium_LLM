# Imports
import os
import logging
from dotenv import load_dotenv

import glob
import numpy as np

from utils import LLM_utils, RAG_utils, FAISS_utils

#%%

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)

#%%
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # numpy issue patch

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not all([OPENAI_API_KEY]):
    raise ValueError("One or more environment variables are missing. Please check your .env file.")


#%%
def create_vector_store(model, processor, device, index, metadata, path="Raw Documents/*"):
    doc_paths = glob.glob(path)

    if not doc_paths:
        print("No documents found in dedicated folder.")
    else:
        print(f"Found {len(doc_paths)} documents to process.")

    for doc_path in doc_paths:
        try:
            text = ""
            images = []

            # Extract text and images based on document type
            file_extension = os.path.splitext(doc_path)[-1].lower()
            if file_extension == ".docx":
                text, images = RAG_utils.extract_text_and_images_from_word(doc_path)
            elif file_extension == ".pdf":
                text, images = RAG_utils.extract_text_and_images_from_pdf(doc_path)
            elif file_extension == ".xlsx":
                text, images = RAG_utils.extract_text_and_images_from_excel(doc_path)
            else:
                print(f"Unsupported file type: {doc_path}")
                continue

            print(f"Processing document: {doc_path}")
            print(f" - Extracted {len(text.split())} words and {len(images)} images.")

            # Encode text in chunks with CLIP
            text_chunks = RAG_utils.chunk_text(text, chunk_size=512) if text.strip() else []
            print(f" - Split text into {len(text_chunks)} chunks for encoding.")

            text_embeddings, _ = LLM_utils.encode_with_clip(text_chunks, [], model, processor, device)
            print(f" - Encoded {len(text_embeddings)} text chunks.")

            # Add text embeddings to FAISS
            for chunk_idx, text_embedding in enumerate(text_embeddings):
                try:
                    content = text_chunks[chunk_idx]
                    FAISS_utils.add_to_faiss(np.array(text_embedding), doc_path, "text-chunk", content, index, metadata)
                    print(f" - Added text chunk {chunk_idx} to FAISS.")
                except Exception as e:
                    print(f"Error adding text embedding for chunk {chunk_idx} in {doc_path}: {e}")

            # Optionally handle images
            # For now, skip adding image embeddings
        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")


#%%
# Initialize CLIP embedding model:
clip_model, clip_processor, device = LLM_utils.CLIP_init()

# Set embedding dimension (default CLIP embedding size is 512)
embedding_dimension = getattr(clip_model, "visual_projection", None)

if embedding_dimension:
    embedding_dimension = embedding_dimension.out_features
else:
    embedding_dimension = 512  # Fallback to standard CLIP embedding size
print(f"Embedding dimension set to: {embedding_dimension}")

# Initialize FAISS index
index = FAISS_utils.initialize_faiss_index(embedding_dimension)

# Initialize FAISS metadata
metadata = []

# Create the vectorstore
create_vector_store(model=clip_model, processor=clip_processor, device=device, index=index, metadata=metadata)

# Save index and metadata
FAISS_utils.save_faiss_index(index, "faiss_index.bin")
FAISS_utils.save_metadata(metadata, "faiss_metadata.json")
