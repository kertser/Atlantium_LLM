"""
This module implements a document processing pipeline for creating and maintaining a
retrieval-augmented generation (RAG) system supported by FAISS and CLIP models.

Key Features:
- Extracts text and images from various document types (PDF, Word, Excel).
- Generates embeddings for text chunks and images using the CLIP model.
- Stores embeddings and metadata in a FAISS index for efficient retrieval.
- Manages stored images and metadata for validation and processing continuity.
- Supports dynamic addition of new documents to the system.
- Implements a basic chat interface for querying the system.

The pipeline is designed to be flexible and can be easily extended to support
additional document types or advanced RAG features as needed.

"""

import sys
import logging
import json
from dotenv import load_dotenv
import glob
import numpy as np
from pathlib import Path
from config import CONFIG
from typing import Any, Tuple, List
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
    """
    Process a list of documents to extract text and images, generate embeddings,
    and store them in a FAISS index.

    Args:
        model: The CLIP model instance for embeddings.
        processor: The CLIP processor for preprocessing.
        device: The device (CPU or GPU) used for processing.
        index: The FAISS index for storing embeddings.
        metadata: A list to store metadata entries corresponding to the embeddings.
        image_store: An ImageStore instance to manage image storage.
        doc_paths: List of paths to documents to process. Defaults to None to process all available documents.

    Returns:
        Tuple containing the updated FAISS index and the generated metadata.
    """
    try:
        # If no specific docs provided, get all documents
        processed_image_ids = set()

        # Track image IDs that have been added to FAISS
        faiss_processed_ids = set()

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
                    text_chunks = chunk_text(text, str(doc_path))  # Fixed: Pass source path
                    if text_chunks:
                        chunk_texts = [chunk['text'] for chunk in text_chunks]  # Get just the text for embedding
                        text_embeddings, _ = encode_with_clip(chunk_texts, [], model, processor, device)

                        for chunk_idx, embedding in enumerate(text_embeddings):
                            if embedding is not None:
                                try:
                                    add_to_faiss(
                                        embedding=np.array(embedding),
                                        source_file_name=doc_path,
                                        content_type="text-chunk",
                                        content=text_chunks[chunk_idx],  # Pass the full chunk dict
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

                                # Add to FAISS with verified image data and duplicate prevention
                                add_to_faiss(
                                    embedding=embedding_to_use,
                                    source_file_name=doc_path,
                                    content_type="image",
                                    content={
                                        "image_id": image_id,
                                        "source_doc": str(doc_path),
                                        "context": img_data.get('context', ''),
                                        "caption": img_data.get('caption', ''),
                                        "page": img_data['page_num'],
                                        "path": str(image_store.images_path / f"{image_id}.png")
                                    },
                                    index=index,
                                    metadata=metadata,
                                    processed_ids=faiss_processed_ids  # Pass set to track processed IDs
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

def get_all_documents(base_path: Path, extensions: List[str]) -> List[Path]:
    """
    Recursively fetch all documents with specified extensions from the base path.

    Args:
        base_path: Path to the base directory to search.
        extensions: List of file extensions to include in the search.

    Returns:
        List of Paths to the matching documents.
    """
    all_docs = []
    for ext in extensions:
        # Use rglob for recursive search
        all_docs.extend([p for p in base_path.rglob(f"*{ext}")])
    return all_docs

def check_stored_images():
    """
    Validate the storage and indexing of images.

    Checks for:
    - Presence of image files in the storage directory.
    - Corresponding metadata entries in the image metadata JSON file.
    - Presence of image entries in the FAISS metadata file.
    """
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
    """
    Update the local record of successfully processed files.

    Args:
        doc_paths: List of paths to documents that were successfully processed.
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
    """
    Identify documents that have not been processed yet.

    Retrieves all available documents and filters out those already listed as processed.

    Returns:
        List of paths to unprocessed documents.
    """
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
    """
    Load the list of documents that have already been processed.

    Returns:
        Set of absolute file paths to processed documents.
    """
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
    Validate and clean both metadata and the FAISS index.

    Ensures metadata integrity and checks the existence of corresponding image data in the ImageStore.

    Args:
        metadata: List of metadata entries to validate.
        index: FAISS index to be cleaned.
        image_store: Instance of ImageStore for validating image entries.

    Returns:
        Tuple containing the cleaned metadata list and the cleaned FAISS index.
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
    """
    Main function to orchestrate the processing pipeline.

    Steps:
    - Load environment variables and configurations.
    - Identify unprocessed documents.
    - Initialize CLIP model and FAISS index.
    - Process documents in batches to extract content, generate embeddings, and update the index.
    - Perform cleanup and save intermediate results.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    try:
        # Load environment variables
        load_dotenv()

        # Get all documents recursively with proper error handling
        try:
            all_docs = get_all_documents(CONFIG.RAW_DOCUMENTS_PATH, CONFIG.SUPPORTED_EXTENSIONS)
            if not all_docs:
                logging.warning("No documents found to process")
                return 0

            # Convert to strings for compatibility
            all_docs = [str(doc) for doc in all_docs]
            logging.info(f"Found {len(all_docs)} documents in total")
        except Exception as e:
            logging.error(f"Error finding documents: {e}")
            return 1

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

            # Create or load FAISS index with proper error handling
            try:
                # Ensure indices directory exists
                CONFIG.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

                if CONFIG.FAISS_INDEX_PATH.exists():
                    try:
                        index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
                        metadata = load_metadata(CONFIG.METADATA_PATH)
                        logging.info("Loaded existing FAISS index and metadata")
                    except Exception as e:
                        logging.warning(f"Failed to load existing index: {e}")
                        index = None

                if index is None:
                    logging.info("Creating new FAISS index")
                    index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
                    metadata = []
                    # Save empty index and metadata immediately
                    save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
                    save_metadata(metadata, CONFIG.METADATA_PATH)

            except Exception as e:
                logging.error(f"Critical error with FAISS initialization: {e}")
                raise

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
                    image_store=ImageStore(CONFIG.STORED_IMAGES_PATH),
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
        sys.exit(result)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
