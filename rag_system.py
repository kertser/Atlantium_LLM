import sys
import logging
import json
from dotenv import load_dotenv
import glob
import numpy as np
from pathlib import Path
from config import CONFIG
from typing import Any, Tuple
import faiss
from utils.FAISS_utils import (
    initialize_faiss_index,
    add_to_faiss,
    save_faiss_index,
    save_metadata,
    load_faiss_index,
    load_metadata
)
from utils.LLM_utils import CLIP_init, encode_with_clip
from utils.RAG_utils import (
    extract_text_and_images_from_pdf,
    extract_text_and_images_from_word,
    extract_text_and_images_from_excel,
    chunk_text,
)
from utils.image_store import (
    ImageStore,
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
        processed_image_ids = set()

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

                            # Skip if we've already processed this image
                            if image_id in processed_image_ids:
                                logging.info(f"Skipping duplicate image {image_id}")
                                continue

                            processed_image_ids.add(image_id)

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
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            print(f"Found {len(metadata)} image entries in metadata")
            for img_id, data in metadata.items():
                print(f"Image {img_id}: {data['source_document']}")
    else:
        print("No image metadata file found")

    # Check FAISS metadata
    if CONFIG.METADATA_PATH.exists():
        with open(CONFIG.METADATA_PATH, 'r', encoding='utf-8') as f:
            faiss_metadata = json.load(f)
            image_entries = [
                m for m in faiss_metadata
                if m.get('type') == 'image' and isinstance(m.get('content'), dict)
                and m['content'].get('image_id')
            ]
            print(f"Found {len(image_entries)} image entries in FAISS metadata")
            # Print details of found images
            for entry in image_entries:
                content = entry['content']
                print(f"FAISS Image {content['image_id']}: {content['source_doc']}")
    else:
        print("No FAISS metadata file found")


def update_processed_files(doc_paths):
    """Update the list of successfully processed files

    Args:
        doc_paths: List of paths to documents that were successfully processed
    """
    processed_files_path = Path("processed_files.json")
    try:
        if processed_files_path.exists():
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()

        absolute_paths = {str(Path(path).absolute()) for path in doc_paths}
        processed_files.update(absolute_paths)

        with open(processed_files_path, 'w', encoding='utf-8') as f:
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
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        logging.error(f"Error loading processed files list: {e}")
        return set()


def validate_metadata_and_index(metadata: list, index: Any, image_store: ImageStore) -> Tuple[list, Any]:
    """
    Validate and clean both metadata and FAISS index.

    Args:
        metadata: List of metadata entries
        index: FAISS index
        image_store: ImageStore instance

    Returns:
        Tuple of (cleaned metadata list, cleaned index)
    """
    valid_metadata = []
    valid_indices = []
    image_ids_processed = set()
    current_idx = 0

    for idx, entry in enumerate(metadata):
        is_valid = False

        if entry.get('type') == 'text-chunk':
            is_valid = True
        elif entry.get('type') == 'image':
            if isinstance(entry.get('content'), dict):
                image_id = entry['content'].get('image_id')
                if image_id and image_id not in image_ids_processed:
                    image, _ = image_store.get_image(image_id)
                    if image is not None:
                        is_valid = True
                        image_ids_processed.add(image_id)

        if is_valid:
            valid_metadata.append(entry)
            valid_indices.append(idx)

    # Create new index with only valid entries
    new_index = faiss.IndexFlatL2(index.d)
    if valid_indices:
        vectors = np.vstack([np.array(index.reconstruct(idx)) for idx in valid_indices])
        new_index.add(vectors)

    return valid_metadata, new_index


def main():
    try:
        # Load environment variables
        load_dotenv()

        # Get list of all documents with proper error handling
        all_docs = []
        for ext in CONFIG.SUPPORTED_EXTENSIONS:
            try:
                docs = glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}"))
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"Error searching for {ext} files: {e}")
                continue

        if not all_docs:
            logging.warning("No documents found to process")
            return 0

        # Load already processed files with error handling
        try:
            processed_files = get_processed_files()
        except Exception as e:
            logging.error(f"Error loading processed files: {e}")
            processed_files = set()

        # Filter out already processed files using absolute paths
        new_docs = [path for path in all_docs if str(Path(path).absolute()) not in processed_files]

        if not new_docs:
            logging.info("All documents have already been processed")
            return 0

        logging.info(f"Found {len(new_docs)} new documents to process")

        # Initialize components with proper cleanup
        clip_model = None
        index = None
        try:
            # Initialize CLIP model
            clip_model, clip_processor, device = CLIP_init(CONFIG.CLIP_MODEL_NAME)
            if not clip_model or not clip_processor:
                raise RuntimeError("Failed to initialize CLIP model")

            # Try to load existing index and metadata
            try:
                index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
                metadata = load_metadata(CONFIG.METADATA_PATH)
                logging.info("Loaded existing FAISS index and metadata")
            except Exception as e:
                logging.info(f"Creating new index and metadata: {e}")
                index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
                metadata = []

            # Initialize ImageStore
            image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)

            # Process documents in smaller batches to manage memory
            batch_size = 5  # Adjust based on available memory
            for i in range(0, len(new_docs), batch_size):
                batch_docs = new_docs[i:i + batch_size]
                logging.info(
                    f"Processing batch {i // batch_size + 1} of {(len(new_docs) + batch_size - 1) // batch_size}")

                updated_index, new_metadata = process_documents(
                    model=clip_model,
                    processor=clip_processor,
                    device=device,
                    index=index,
                    metadata=metadata,
                    image_store=image_store,
                    doc_paths=batch_docs
                )

                # Update index and metadata after each batch
                if new_metadata:
                    index = updated_index
                    metadata.extend(new_metadata)

                    # Save intermediate results
                    save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
                    save_metadata(metadata, CONFIG.METADATA_PATH)
                    update_processed_files(batch_docs)

                    logging.info(f"Saved progress after batch {i // batch_size + 1}")

            logging.info("Processing completed successfully")
            return 0

        except Exception as e:
            logging.error(f"Critical error during processing: {str(e)}", exc_info=True)
            return 1

        finally:
            # Cleanup
            if clip_model is not None and hasattr(clip_model, 'cpu'):
                try:
                    clip_model.cpu()  # Move model to CPU to free GPU memory
                    del clip_model
                except Exception as e:
                    logging.error(f"Error cleaning up CLIP model: {e}")

            if index is not None:
                try:
                    del index
                except Exception as e:
                    logging.error(f"Error cleaning up FAISS index: {e}")

            import gc
            gc.collect()  # Force garbage collection

    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        result = main()
        # After processing, check the stored images
        check_stored_images()
        sys.exit(result)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
