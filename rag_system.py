import os,sys
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
    chunk_text,
    get_relevant_images,
    extract_text_around_image
)

from image_store import (
    ImageStore,
    deduplicate_images,
    remove_duplicate_images,
    update_faiss_metadata
)

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


def process_documents(model, processor, device, index, metadata, image_store, doc_paths=None):
    """Process documents and store their content and images.

    Args:
        model: CLIP model instance
        processor: CLIP processor instance
        device: Computing device (cpu/cuda)
        index: FAISS index instance
        metadata: List of metadata entries
        image_store: ImageStore instance
        doc_paths: Optional list of specific documents to process. If None, processes all documents.
    """
    try:
        # If no specific docs provided, get all documents
        if doc_paths is None:
            doc_paths = []
            for ext in CONFIG.SUPPORTED_EXTENSIONS:
                doc_paths.extend(glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}")))

        if not doc_paths:
            logging.warning("No documents found to process")
            return index, metadata

        logging.info(f"Found {len(doc_paths)} documents to process")

        # Track successful embeddings
        embeddings_added = False

        # Process each document
        for doc_path in doc_paths:
            try:
                text = ""
                images_data = []
                file_extension = Path(doc_path).suffix.lower()

                # Extract content based on file type
                if file_extension == '.pdf':
                    text, images_data = extract_text_and_images_from_pdf(doc_path)
                elif file_extension == '.docx':
                    text, images_data = extract_text_and_images_from_word(doc_path)
                elif file_extension == '.xlsx':
                    text, images_data = extract_text_and_images_from_excel(doc_path)

                # Process text content
                if text and text.strip():
                    text_chunks = chunk_text(text, CONFIG.CHUNK_SIZE)
                    if text_chunks:
                        text_embeddings, _ = encode_with_clip(text_chunks, [], model, processor, device)

                        for chunk_idx, embedding in enumerate(text_embeddings):
                            if embedding is not None:
                                try:
                                    add_to_faiss(
                                        embedding=np.array(embedding),
                                        pdf_name=doc_path,
                                        content_type="text-chunk",
                                        content=text_chunks[chunk_idx],
                                        index=index,
                                        metadata=metadata
                                    )
                                    embeddings_added = True
                                    logging.info(f"Added text chunk {chunk_idx} from {doc_path}")
                                except Exception as e:
                                    logging.error(f"Error adding text chunk {chunk_idx}: {e}")

                # Process images
                if images_data:
                    for img_data in images_data:
                        try:
                            # Store image first and ensure it's stored successfully
                            image_id = image_store.store_image(
                                image=img_data['image'],
                                source_doc=str(doc_path),
                                page_num=img_data['page_num'],
                                context=img_data.get('context', ''),
                                caption=img_data.get('caption', f"Image from {Path(doc_path).name}")
                            )

                            # Verify image was stored successfully
                            stored_image, _ = image_store.get_image(image_id)
                            if stored_image is None:
                                logging.error(f"Failed to verify stored image {image_id}")
                                continue

                            # Create image embedding
                            _, image_embedding = encode_with_clip([], [img_data['image']], model, processor, device)

                            if isinstance(image_embedding, np.ndarray) and image_embedding.size > 0:
                                if len(image_embedding.shape) > 1:
                                    embedding_to_use = image_embedding[0]
                                else:
                                    embedding_to_use = image_embedding

                                # Add to FAISS with verified image data
                                add_to_faiss(
                                    embedding=embedding_to_use,
                                    pdf_name=doc_path,
                                    content_type="image",
                                    content={
                                        "image_id": image_id,
                                        "source_doc": str(doc_path),
                                        "context": img_data.get('context', ''),
                                        "caption": img_data.get('caption', ''),
                                        "page": img_data['page_num'],
                                        "path": str(image_store.images_path / f"{image_id}.png")  # Add actual path
                                    },
                                    index=index,
                                    metadata=metadata
                                )
                                embeddings_added = True
                                logging.info(f"Added image embedding for {image_id}")
                        except Exception as e:
                            logging.error(f"Error processing image from {doc_path}: {e}")
                            continue

                # After processing, verify image accessibility
                image_ids = [m.get('content', {}).get('image_id') for m in metadata
                             if m.get('type') == 'image' and isinstance(m.get('content'), dict)]

                for image_id in image_ids:
                    if image_id:
                        image, _ = image_store.get_image(image_id)
                        if image is None:
                            logging.warning(f"Verification failed for image {image_id}")

            except Exception as e:
                logging.error(f"Error processing document {doc_path}: {e}")
                continue

        if not embeddings_added:
            logging.warning("No embeddings were added during processing")
            return index, []

        if not metadata:
            logging.warning("No metadata generated during processing")
            return index, []

        logging.info(f"Successfully processed {len(doc_paths)} documents")
        logging.info(f"Generated {len(metadata)} metadata entries")
        return index, metadata

    except Exception as e:
        logging.error(f"Error during document processing: {e}")
        raise


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


def update_processed_files(doc_paths):
    """Update the list of successfully processed files

    Args:
        doc_paths: List of paths to documents that were successfully processed
    """
    processed_files_path = Path("processed_files.json")
    try:
        # Load existing processed files
        if processed_files_path.exists():
            with open(processed_files_path, 'r') as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()

        # Convert paths to absolute paths and add to set
        absolute_paths = {str(Path(path).absolute()) for path in doc_paths}
        processed_files.update(absolute_paths)

        # Save updated list
        with open(processed_files_path, 'w') as f:
            json.dump(list(processed_files), f, indent=2)

        logging.info(f"Updated processed files list with {len(doc_paths)} new documents")

    except Exception as e:
        logging.error(f"Error updating processed files list: {e}")
        raise


def get_unprocessed_documents():
    """Get list of documents that haven't been processed yet"""
    try:
        # Get all documents in the raw documents directory
        all_docs = []
        for ext in CONFIG.SUPPORTED_EXTENSIONS:
            all_docs.extend(glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}")))

        # Get list of processed files
        processed_files = get_processed_files()

        # Filter out processed files
        unprocessed = [doc for doc in all_docs
                       if str(Path(doc).absolute()) not in processed_files]

        return unprocessed
    except Exception as e:
        logging.error(f"Error getting unprocessed documents: {e}")
        return []


# In rag_system.py, update the main function

def get_processed_files():
    """Load list of already processed files"""
    processed_files_path = Path("processed_files.json")
    try:
        if processed_files_path.exists():
            with open(processed_files_path, 'r') as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        logging.error(f"Error loading processed files list: {e}")
        return set()


def main():
    try:
        # Load environment variables
        load_dotenv()

        # Get list of all documents
        all_docs = []
        for ext in CONFIG.SUPPORTED_EXTENSIONS:
            all_docs.extend(glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}")))

        if not all_docs:
            logging.warning("No documents found to process")
            return 0

        # Load already processed files
        processed_files = get_processed_files()

        # Filter out already processed files using absolute paths
        new_docs = [path for path in all_docs
                   if str(Path(path).absolute()) not in processed_files]

        if not new_docs:
            logging.info("All documents have already been processed")
            return 0

        logging.info(f"Found {len(new_docs)} new documents to process")

        # Initialize CLIP model
        clip_model, clip_processor, device = CLIP_init(CONFIG.CLIP_MODEL_NAME)
        if not clip_model or not clip_processor:
            raise RuntimeError("Failed to initialize CLIP model")

        # Try to load existing index and metadata first
        try:
            index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
            metadata = load_metadata(CONFIG.METADATA_PATH)
            logging.info("Loaded existing FAISS index and metadata")
        except Exception as e:
            logging.info("No existing index found or error loading. Initializing new ones")
            index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
            metadata = []

        # Initialize ImageStore
        image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)

        # Process only new documents
        updated_index, new_metadata = process_documents(
            model=clip_model,
            processor=clip_processor,
            device=device,
            index=index,
            metadata=metadata,
            image_store=image_store,
            doc_paths=new_docs  # Pass only new documents
        )

        if not new_metadata and not metadata:  # Check both new and existing metadata
            logging.warning("No metadata generated during processing")
            # Save empty index and metadata to indicate processing completed
            save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
            save_metadata([], CONFIG.METADATA_PATH)
            return 0

        # If processing was successful, update metadata and save
        if new_metadata:
            if isinstance(metadata, list):
                metadata.extend(new_metadata)
            else:
                metadata = new_metadata

            # Save updated results
            save_faiss_index(updated_index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(metadata, CONFIG.METADATA_PATH)

            # Update processed files list
            update_processed_files(new_docs)

            logging.info(f"Successfully added {len(new_metadata)} new entries to metadata")
            logging.info("Processing completed successfully")
            return 0
        else:
            logging.warning("No new metadata generated, keeping existing data")
            return 0

    except Exception as e:
        logging.error(f"Critical error during processing: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    result = main()
    # After processing, check the stored images
    check_stored_images()
    sys.exit(result)