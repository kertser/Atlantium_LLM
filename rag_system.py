import os
import logging
import json
from dotenv import load_dotenv
import glob
import numpy as np
from pathlib import Path

from config import CONFIG
from utils.FAISS_utils import initialize_faiss_index, add_to_faiss, save_faiss_index, save_metadata
from utils.LLM_utils import CLIP_init, encode_with_clip
from utils.RAG_utils import (
    extract_text_and_images_from_pdf,
    extract_text_and_images_from_word,
    extract_text_and_images_from_excel,
    chunk_text
)
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

def process_document_images(images_data, doc_path, image_store):
    """Helper function to process and store images with context from a document"""
    image_ids = []
    for idx, img_data in enumerate(images_data):
        try:
            image_id = image_store.store_image(
                image=img_data['image'],
                source_doc=str(doc_path),
                page_num=img_data['page_num'],
                caption=f"Image {idx+1} from {Path(doc_path).name}",
                context=img_data['context']
            )
            image_ids.append(image_id)
            logging.info(f"Stored image {idx+1} from {doc_path} with ID: {image_id}")
        except Exception as e:
            logging.error(f"Error storing image {idx+1} from {doc_path}: {e}")
    return image_ids

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
            image_ids = []
            file_extension = Path(doc_path).suffix.lower()

            # Extract text and images based on document type
            logging.info(f"Processing document: {doc_path}")

            if file_extension == '.pdf':
                text, images = extract_text_and_images_from_pdf(doc_path)
                image_ids.extend(process_document_images(images, doc_path, image_store))
            elif file_extension == '.docx':
                text, images = extract_text_and_images_from_word(doc_path)
                image_ids.extend(process_document_images(images, doc_path, image_store))
            elif file_extension == '.xlsx':
                text, images = extract_text_and_images_from_excel(doc_path)
                image_ids.extend(process_document_images(images, doc_path, image_store))
            else:
                logging.warning(f"Unsupported file type: {file_extension}")
                continue

            logging.info(f"Extracted {len(text.split())} words and {len(image_ids)} images from {doc_path}")

            # Process text content
            if text.strip():
                text_chunks = chunk_text(text, CONFIG.CHUNK_SIZE)
                logging.info(f"Split text into {len(text_chunks)} chunks for encoding.")

                text_embeddings, _ = encode_with_clip(text_chunks, [], model, processor, device)
                logging.info(f"Encoded {len(text_embeddings)} text chunks.")

                # Add text embeddings to FAISS
                for chunk_idx, text_embedding in enumerate(text_embeddings):
                    try:
                        content = text_chunks[chunk_idx]
                        add_to_faiss(np.array(text_embedding), doc_path, "text-chunk", content, index, metadata)
                    except Exception as e:
                        logging.error(f"Error adding text embedding for chunk {chunk_idx} in {doc_path}: {e}")

            # Add image embeddings to FAISS
            if image_ids:
                for img_id in image_ids:
                    try:
                        image, image_metadata = image_store.get_image(img_id)
                        if image:
                            _, image_embedding = encode_with_clip([], [image], model, processor, device)
                            if len(image_embedding) > 0:
                                metadata_entry = {
                                    "type": "image",
                                    "image_id": img_id,
                                    "source_doc": str(doc_path),
                                    "caption": image_metadata.get("caption", ""),
                                    "page": image_metadata.get("page_number", 0)
                                }
                                add_to_faiss(
                                    embedding=np.array(image_embedding[0]),
                                    pdf_name=doc_path,
                                    content_type="image",
                                    content=metadata_entry,
                                    index=index,
                                    metadata=metadata
                                )
                                logging.info(f"Added image {img_id} to FAISS index")
                    except Exception as e:
                        logging.error(f"Error processing image {img_id} from {doc_path}: {e}")

        except Exception as e:
            logging.error(f"Error processing document {doc_path}: {e}")
            continue

    return index, metadata

def check_stored_images():
    """Check if images are properly stored and indexed"""
    # Check physical image files
    images_path = CONFIG.STORED_IMAGES_PATH / "images"
    if not images_path.exists():
        print(f"Images directory not found at {images_path}")
        return

    stored_images = list(images_path.glob("*.png"))
    print(f"Found {len(stored_images)} stored image files")

    # Check metadata file
    metadata_path = CONFIG.STORED_IMAGES_PATH / "image_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"Found {len(metadata)} image entries in metadata")
            for img_id, data in metadata.items():
                print(f"Image {img_id}: {data['source_document']}")
    else:
        print("No image metadata file found")

    # Check FAISS metadata
    if CONFIG.METADATA_PATH.exists():
        with open(CONFIG.METADATA_PATH, 'r') as f:
            faiss_metadata = json.load(f)
            image_entries = [m for m in faiss_metadata if m.get('type') == 'image']
            print(f"Found {len(image_entries)} image entries in FAISS metadata")
    else:
        print("No FAISS metadata file found")

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
    index, metadata = process_documents(
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
    # After processing, check the stored images
    check_stored_images()