import os
import logging
from dotenv import load_dotenv
import glob
import numpy as np
from pathlib import Path

from config import CONFIG
from utils.FAISS_utils import initialize_faiss_index, add_to_faiss, save_faiss_index, save_metadata
from utils.LLM_utils import CLIP_init, encode_with_clip
from utils.RAG_utils import extract_text_and_images_from_pdf, extract_text_and_images_from_word, \
    extract_text_and_images_from_excel, chunk_text
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


def process_documents(model, processor, device, index, metadata, image_store):
    """Process all documents in the Raw Documents folder and add them to the RAG system."""
    # Get all documents from the Raw Documents folder
    doc_paths = []
    for ext in CONFIG.SUPPORTED_EXTENSIONS:
        doc_paths.extend(glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}")))

    if not doc_paths:
        logging.warning(f"No documents found in {CONFIG.RAW_DOCUMENTS_PATH}")
        return

    logging.info(f"Found {len(doc_paths)} documents to process.")

    for doc_path in doc_paths:
        try:
            text = ""
            image_ids = []  # Store image IDs instead of images
            file_extension = Path(doc_path).suffix.lower()

            # Extract text and images based on document type
            logging.info(f"Processing document: {doc_path}")

            if file_extension == '.pdf':
                text, image_ids = extract_text_and_images_from_pdf(doc_path, image_store)
            elif file_extension == '.docx':
                text, image_ids = extract_text_and_images_from_word(doc_path, image_store)
            elif file_extension == '.xlsx':
                text, image_ids = extract_text_and_images_from_excel(doc_path, image_store)
            else:
                logging.warning(f"Unsupported file type: {doc_path}")
                continue

            logging.info(f" - Extracted {len(text.split())} words and {len(image_ids)} images.")

            # Process text content
            if text.strip():
                text_chunks = chunk_text(text, CONFIG.CHUNK_SIZE)
                logging.info(f" - Split text into {len(text_chunks)} chunks for encoding.")

                text_embeddings, _ = encode_with_clip(text_chunks, [], model, processor, device)
                logging.info(f" - Encoded {len(text_embeddings)} text chunks.")

                # Add text embeddings to FAISS
                for chunk_idx, text_embedding in enumerate(text_embeddings):
                    try:
                        content = text_chunks[chunk_idx]
                        add_to_faiss(np.array(text_embedding), doc_path, "text-chunk", content, index, metadata)
                    except Exception as e:
                        logging.error(f"Error adding text embedding for chunk {chunk_idx} in {doc_path}: {e}")

            # Add image embeddings to FAISS
            if image_ids:
                for image_id in image_ids:
                    try:
                        image, image_metadata = image_store.get_image(image_id)
                        if image:
                            _, image_embedding = encode_with_clip([], [image], model, processor, device)
                            if len(image_embedding) > 0:
                                add_to_faiss(
                                    embedding=np.array(image_embedding[0]),
                                    pdf_name=doc_path,
                                    content_type="image",
                                    content=f"Image {image_id}",
                                    index=index,
                                    metadata=metadata
                                )
                                # Add image_id to metadata for retrieval
                                metadata[-1]['image_id'] = image_id
                    except Exception as e:
                        logging.error(f"Error processing image {image_id} from {doc_path}: {e}")

        except Exception as e:
            logging.error(f"Error processing document {doc_path}: {e}")
            continue


def main():
    # Load environment variables
    load_dotenv()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # numpy issue patch

    # Initialize CLIP embedding model
    clip_model, clip_processor, device = CLIP_init(CONFIG.CLIP_MODEL_NAME)
    logging.info(f"CLIP model initialized on {device}")

    # Initialize FAISS index and metadata
    index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
    metadata = []

    # Initialize ImageStore
    image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)
    logging.info("ImageStore initialized")

    # Process documents
    process_documents(
        model=clip_model,
        processor=clip_processor,
        device=device,
        index=index,
        metadata=metadata,
        image_store=image_store
    )

    # Save index and metadata
    save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
    save_metadata(metadata, CONFIG.METADATA_PATH)
    logging.info("Index and metadata saved successfully")


if __name__ == "__main__":
    main()