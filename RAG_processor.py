import glob
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional, Union
import gc
import torch

import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from config import CONFIG
from utils.FAISS_utils import (
    initialize_faiss_index,
    add_to_faiss,
    save_faiss_index,
    save_metadata,
    load_faiss_index,
    load_metadata,
    optimize_faiss_index,
)
from utils.LLM_utils import CLIP_init, encode_with_clip
from utils.RAG_utils import (
    extract_text_and_images_from_pdf,
    extract_text_and_images_from_word,
    extract_text_and_images_from_excel,
    chunk_text,
)
# Updated import: combine image utilities into one module
from utils.img_utils import ImageStore, ImageClassifier, ImageProcessor

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.LOG_PATH / "system.log")
    ]
)

def filter_technical_images(images_data, model, processor, device, source_doc):
    filtered_images = []
    classifier = ImageClassifier(model=model, processor=processor, device=device)
    labels = ["a technical image", "a non-technical image"]

    logging.info(f"Processing {len(images_data)} images for technical content")  # Added logging

    for img_data in images_data:
        try:
            image = img_data['image']
            logging.info(f"Image size: {image.size}, mode: {image.mode}")  # Added logging

            # Validate image before processing - silently skip invalid images
            if not ImageProcessor.validate_image_data(image):
                continue

            # Convert to RGB safely
            try:
                image = ImageProcessor.convert_to_rgb(image)
            except Exception as e:
                logger.debug(f"Error converting image format: {e}")  # Changed to debug
                continue

            # Perform classification
            try:
                predicted_label, confidence = classifier.classify(
                    image=image,
                    labels=labels
                )
            except Exception as e:
                logger.debug(f"Classification error: {e}")  # Changed to debug
                continue

            similarity = confidence if predicted_label == "a technical image" else 1 - confidence

            if similarity > CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD:
                img_data['technical_similarity'] = similarity
                filtered_images.append(img_data)
                logger.debug(  # Changed to debug
                    f"Technical image found in {source_doc} "
                    f"(similarity: {similarity:.4f})"
                )

        except Exception as e:
            logger.debug(f"Error processing image from {source_doc}: {e}")  # Changed to debug
            continue

    return filtered_images



def compress_metadata(metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove unnecessary fields and compress metadata.

    Args:
        metadata: List of metadata dictionaries to compress

    Returns:
        List of compressed metadata dictionaries
    """
    compressed: List[Dict[str, Any]] = []
    for entry in metadata:
        if not isinstance(entry, dict):
            logging.warning(f"Skipping invalid metadata entry: {entry}")
            continue

        # Keep only essential fields
        minimal_entry: Dict[str, Any] = {
            'type': entry.get('type', 'unknown'),
            'source_file_name': entry.get('source_file_name', ''),
            'id': entry.get('id', '')
        }

        # Handle content field safely
        content = entry.get('content', {})
        if isinstance(content, dict):
            if entry.get('type') == 'text-chunk':
                minimal_entry['content'] = {
                    'text': content.get('text', '')[:CONFIG.METADATA_TEXT_LIMIT],  # Limit text size
                    'metadata': content.get('metadata', {})
                }
            elif entry.get('type') == 'image':
                minimal_entry['content'] = {
                    'image_id': content.get('image_id', ''),
                    'source_doc': content.get('source_doc', ''),
                    'page': content.get('page', 0)
                }
        else:
            minimal_entry['content'] = {}

        compressed.append(minimal_entry)
    return compressed


def cleanup_metadata(metadata, index):
    """Remove duplicate entries and validate existing ones"""
    if not metadata or not isinstance(metadata, list):
        logging.warning("Empty or invalid metadata, returning empty list")
        return [], index

    seen_hashes = set()
    cleaned_metadata = []
    valid_indices = []

    for idx, entry in enumerate(metadata):
        try:
            if not isinstance(entry, dict) or 'content' not in entry:
                continue

            content = entry.get('content', {})
            if isinstance(content, dict):
                content_str = json.dumps(content, sort_keys=True, default=str)
                content_hash = hashlib.md5(content_str.encode()).hexdigest()

                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    cleaned_metadata.append(entry)
                    valid_indices.append(idx)
        except Exception as e:
            logging.warning(f"Error processing metadata entry {idx}: {e}")
            continue

    # Rebuild index with only valid entries
    try:
        if index is not None and valid_indices:
            new_index = faiss.IndexFlatL2(index.d)
            vectors = np.vstack([index.reconstruct(idx) for idx in valid_indices])
            new_index.add(vectors)
            return cleaned_metadata, new_index
    except Exception as e:
        logging.error(f"Error rebuilding index: {e}")
        return cleaned_metadata, index

    return cleaned_metadata, index


def process_incrementally(docs: List[Path], batch_size: int = CONFIG.BATCH_SIZE) -> None:
    """Process documents incrementally and save progress"""
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            # Process batch
            process_documents(batch)
            # Save progress
            update_processed_files(batch)
            # Cleanup after each batch
            metadata, index = cleanup_metadata(load_metadata(), load_faiss_index())
            save_metadata(metadata)
            save_faiss_index(index)
        except Exception as e:
            logging.error(f"Error processing batch {i // batch_size}: {e}")
            continue


def process_documents(
    model: Any,
    processor: Any,
    device: str,
    index: Any,
    metadata: List[Dict],
    image_store: ImageStore,
    doc_paths: Optional[List[str]] = None
) -> Tuple[Any, List[Dict]]:
    """
    Process documents to extract text and technical images, generate embeddings,
    and store them in FAISS index.
    """
    try:
        processed_image_ids = set()
        faiss_processed_ids = set()
        embeddings_added = False
        chunks_processed = set()

        # Load existing metadata if metadata list is empty
        if not metadata and CONFIG.METADATA_PATH.exists():
            try:
                with open(CONFIG.METADATA_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logging.info(f"Loaded {len(metadata)} existing metadata entries")
            except Exception as e:
                logging.warning(f"Could not load existing metadata: {e}")
                metadata = []

        if doc_paths is None:
            doc_paths = []
            for ext in CONFIG.SUPPORTED_EXTENSIONS:
                doc_paths.extend(glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}")))

        if not doc_paths:
            logging.warning("No documents found to process")
            return index, metadata

        logging.info(f"Found {len(doc_paths)} documents to process")

        with tqdm(total=len(doc_paths), desc="Processing documents", unit="doc") as pbar:
            for doc_path in doc_paths:
                try:
                    text = ""
                    images_data = []
                    file_extension = Path(doc_path).suffix.lower()
                    doc_path = Path(doc_path)

                    # Extract content based on file type
                    if file_extension == '.pdf':
                        text, images_data = extract_text_and_images_from_pdf(doc_path)
                    elif file_extension == '.docx':
                        text, images_data = extract_text_and_images_from_word(doc_path)
                    elif file_extension == '.xlsx':
                        text, images_data = extract_text_and_images_from_excel(doc_path)

                    # Process text chunks first
                    if text and text.strip():
                        text_chunks = chunk_text(text, str(doc_path))
                        if text_chunks:
                            chunk_texts = [chunk['text'] for chunk in text_chunks]
                            text_embeddings, _ = encode_with_clip(chunk_texts, [], model, processor, device)

                            for chunk_idx, embedding in enumerate(text_embeddings):
                                if embedding is not None:
                                    chunk_hash = hashlib.md5(chunk_texts[chunk_idx].encode()).hexdigest()

                                    # Check if chunk already exists in metadata
                                    chunk_exists = False
                                    for entry in metadata:
                                        if (entry.get('type') == 'text-chunk' and
                                                entry.get('chunk_hash') == chunk_hash and
                                                entry.get('path') == str(doc_path)):
                                            chunk_exists = True
                                            break

                                    if not chunk_exists and chunk_hash not in chunks_processed:
                                        chunks_processed.add(chunk_hash)
                                        add_to_faiss(
                                            embedding=np.array(embedding),
                                            source_file_name=str(doc_path),
                                            content_type="text-chunk",
                                            content={
                                                'text': chunk_texts[chunk_idx],
                                                'metadata': text_chunks[chunk_idx].get('metadata', {}),
                                                'chunk_hash': chunk_hash  # Add hash to content
                                            },
                                            index=index,
                                            metadata=metadata
                                        )
                                        embeddings_added = True

                    # Filter and process images
                    if images_data:
                        logging.info(f"Processing {len(images_data)} images from {doc_path}")
                        # Add size check before classification
                        images_data = [
                            img for img in images_data
                            if img['image'].width >= CONFIG.MIN_IMAGE_SIZE
                               and img['image'].height >= CONFIG.MIN_IMAGE_SIZE
                        ]
                        filtered_images = filter_technical_images(
                            images_data, model, processor, device, str(doc_path)
                        )

                        for img_data in filtered_images:
                            try:
                                # Store image and get ID
                                image_id = image_store.store_image(
                                    image=img_data['image'],
                                    source_doc=str(doc_path),
                                    page_num=img_data['page_num'],
                                    context=img_data.get('context', ''),
                                    caption=img_data.get('caption', '')
                                )

                                if image_id not in processed_image_ids:
                                    processed_image_ids.add(image_id)

                                    # Generate image embedding
                                    text_embeddings, image_embeddings = encode_with_clip(
                                        [], [img_data['image']],
                                        model, processor, device
                                    )

                                    # Process image embeddings if available
                                    if image_embeddings is not None and len(image_embeddings) > 0:
                                        embedding_to_use = image_embeddings[0] if len(
                                            image_embeddings.shape) > 1 else image_embeddings

                                        # Add to FAISS index
                                        add_to_faiss(
                                            embedding=embedding_to_use,
                                            source_file_name=str(doc_path),
                                            content_type="image",
                                            content={
                                                "image_id": image_id,
                                                "source_doc": str(doc_path),
                                                "context": img_data.get('context', ''),
                                                "caption": img_data.get('caption', ''),
                                                "page": img_data['page_num']
                                            },
                                            index=index,
                                            metadata=metadata,
                                            processed_ids=faiss_processed_ids
                                        )
                                        embeddings_added = True
                                        logging.info(f"Added image {image_id} to FAISS index")
                                    else:
                                        logging.error(f"Failed to generate embedding for image {image_id}")

                            except Exception as e:
                                logging.error(f"Error processing image from {doc_path}: {str(e)}")
                                continue

                    pbar.update(1)

                except Exception as e:
                    logging.error(f"Error processing document {doc_path}: {str(e)}")
                    continue

        # Save index and metadata if any embeddings were added
        if embeddings_added:
            save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(metadata, CONFIG.METADATA_PATH)
            logging.info(f"Saved updated FAISS index with {len(metadata)} entries")

        update_processed_files(doc_paths)
        return index, metadata

    except Exception as e:
        logging.error(f"Error during document processing: {str(e)}")
        raise


def get_all_documents(
    base_path: Path = CONFIG.RAW_DOCUMENTS_PATH,
    extensions: List[str] = CONFIG.SUPPORTED_EXTENSIONS
) -> List[Path]:
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
    if not CONFIG.STORED_IMAGES_PATH.exists():
        print(f"Images directory not found at {CONFIG.STORED_IMAGES_PATH}")
        return

    stored_images = list(CONFIG.STORED_IMAGES_PATH.glob(f"*.{CONFIG.PREFERRED_SAVE_FORMAT.lower()}"))
    print(f"Found {len(stored_images)} stored image files")

    # Use CONFIG path for metadata
    if CONFIG.IMAGE_METADATA_PATH.exists():
        with open(CONFIG.IMAGE_METADATA_PATH, 'r', encoding='utf-8') as f:
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
            for entry in image_entries:
                content = entry['content']
                print(f"FAISS Image {content['image_id']}: {content['source_doc']}")
    else:
        print("No FAISS metadata file found")


def update_processed_files(doc_paths: List[Union[str, Path]]) -> None:
    """Update the local record of successfully processed files."""
    processed_files_path = CONFIG.BASE_DIR / "processed_files.json"  # Changed to base dir
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
        all_docs = []
        for ext in CONFIG.SUPPORTED_EXTENSIONS:
            all_docs.extend(glob.glob(str(CONFIG.RAW_DOCUMENTS_PATH / f"*{ext}")))

        processed_files = get_processed_files()
        unprocessed = [doc for doc in all_docs
                       if str(Path(doc).absolute()) not in processed_files]

        return unprocessed
    except Exception as e:
        logging.error(f"Error getting unprocessed documents: {e}")
        return []


def get_processed_files():
    """Load the list of documents that have already been processed."""
    processed_files_path = CONFIG.BASE_DIR / "processed_files.json"  # Changed to base dir
    try:
        if processed_files_path.exists():
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        logging.error(f"Error loading processed files list: {e}")
        return set()


def validate_metadata_entry(entry):
    """Validate the structure of a metadata entry"""
    if not isinstance(entry, dict):
        return False

    required_fields = ['type', 'source_file_name', 'content']
    if not all(field in entry for field in required_fields):
        return False

    if not isinstance(entry['content'], dict):
        return False

    return True


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
    """Main function with progress bars for batch processing and optimized memory management."""
    clip_model = None
    clip_processor = None
    index = None
    metadata = []

    try:
        load_dotenv()

        # Initialize document discovery
        all_docs = get_all_documents(CONFIG.RAW_DOCUMENTS_PATH, CONFIG.SUPPORTED_EXTENSIONS)
        if not all_docs:
            logging.warning("No documents found to process")
            return 0

        all_docs = [str(doc) for doc in all_docs]
        logging.info(f"Found {len(all_docs)} documents in total")

        # Get processed files
        processed_files = get_processed_files()
        new_docs = [path for path in all_docs if str(Path(path).absolute()) not in processed_files]

        if not new_docs:
            logging.info("All documents have already been processed")
            return 0

        logging.info(f"Found {len(new_docs)} new documents to process")

        # Initialize CLIP and FAISS
        with tqdm(desc="Initializing", total=2) as init_pbar:
            # Initialize CLIP
            clip_model, clip_processor, device = CLIP_init(CONFIG.CLIP_MODEL_NAME)
            if not clip_model or not clip_processor:
                raise RuntimeError("Failed to initialize CLIP model")
            init_pbar.update(1)

            # Initialize FAISS
            CONFIG.RAG_DATA.mkdir(parents=True, exist_ok=True)

            # Load or create FAISS index
            if CONFIG.FAISS_INDEX_PATH.exists():
                try:
                    index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
                    metadata = load_metadata(CONFIG.METADATA_PATH)

                    # Validate and cleanup existing metadata
                    metadata, index = cleanup_metadata(metadata, index)
                    metadata = compress_metadata(metadata)
                    index, metadata = optimize_faiss_index(index, metadata)
                    logging.info("Loaded and optimized existing FAISS index and metadata")
                except Exception as e:
                    logging.warning(f"Failed to load existing index: {e}")
                    index = None

            if index is None:
                logging.info("Creating new FAISS index")
                index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
                metadata = []
                save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
                save_metadata(metadata, CONFIG.METADATA_PATH)

            init_pbar.update(1)

        # Process documents in batches
        batch_size = CONFIG.BATCH_SIZE
        num_batches = (len(new_docs) + batch_size - 1) // batch_size
        with ImageStore() as image_store:
            with tqdm(total=num_batches, desc="Processing batches", unit="batch") as batch_pbar:
                for i in range(0, len(new_docs), batch_size):
                    batch_docs = new_docs[i:i + batch_size]
                    current_batch = (i // batch_size) + 1
                    batch_pbar.set_postfix({
                        "Batch": f"{current_batch}/{num_batches}",
                        "Files": len(batch_docs)
                    })

                    try:
                        updated_index, new_metadata = process_documents(
                            model=clip_model,
                            processor=clip_processor,
                            device=device,
                            index=index,
                            metadata=metadata,
                            image_store=image_store,
                            doc_paths=batch_docs
                        )

                        if new_metadata:
                            valid_new_metadata = [
                                entry for entry in new_metadata
                                if validate_metadata_entry(entry)
                            ]

                            if valid_new_metadata:
                                index = updated_index
                                metadata.extend(valid_new_metadata)

                                # Periodic optimization
                                if i > 0 and i % (batch_size * CONFIG.CLEANUP_FREQUENCY) == 0:
                                    logging.info("Performing periodic optimization...")
                                    metadata, index = cleanup_metadata(metadata, index)
                                    metadata = compress_metadata(metadata)
                                    index, metadata = optimize_faiss_index(index, metadata)

                                # Save progress
                                save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
                                save_metadata(metadata, CONFIG.METADATA_PATH)
                                update_processed_files(batch_docs)

                    except Exception as e:
                        logging.error(f"Error processing batch {current_batch}: {e}", exc_info=True)
                        continue
                    finally:
                        batch_pbar.update(1)
                        gc.collect()

        logging.info("Processing completed successfully")
        return 0

    except Exception as e:
        logging.error(f"Critical error during processing: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup resources
        try:
            if 'image_store' in locals() and image_store is not None:
                image_store.cleanup()
            if clip_model is not None and hasattr(clip_model, 'cpu'):
                clip_model.cpu()
                del clip_model
            if clip_processor is not None:
                del clip_processor
            if index is not None:
                del index
            # Cleanup temporary files
            tmp_files = list(CONFIG.RAG_DATA.glob("*.tmp"))
            for tmp_file in tmp_files:
                try:
                    tmp_file.unlink()
                except Exception as e:
                    logging.error(f"Error removing temporary file {tmp_file}: {e}")
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error during cleanup: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(result)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
