import glob
import hashlib
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional, Union
import gc
import torch
import psutil

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
from utils.img_utils import ImageStore, ImageProcessor
from utils.helper_functions import cleanup_and_reload_resources
from contextlib import contextmanager


def setup_logger():
    """
    Setup internal logging configuration with separated file and console output levels
    """
    # Disable the UserWarning from Flash-Attention (GPU capabilities >8.0)
    warnings.simplefilter("ignore", UserWarning)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # File handler - for all logs (INFO and above)
    file_handler = logging.FileHandler(CONFIG.LOG_PATH / "system.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler - for warnings and errors only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Base level for the logger

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Optional: Add a debug log file for detailed logging
    debug_handler = logging.FileHandler(CONFIG.LOG_PATH / "debug.log")
    debug_handler.setLevel(logging.ERROR)
    debug_handler.setFormatter(file_formatter)
    root_logger.addHandler(debug_handler)


def get_optimal_processing_params():
    """ Initially for multiprocessing """
    # Get system info
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    else:
        gpu_mem = 0

    # Calculate optimal values
    max_concurrent = max(2, min(cpu_count - 2, 12))  # Leave 2 cores for system

    if gpu_mem > 0:
        # GPU available
        max_batch = int(min(64, gpu_mem * 4))  # 4 samples per GB of VRAM
        batch_size = max(8, max_batch // 2)  # Half of max batch size
        min_batch = max(4, batch_size // 4)  # Quarter of batch size
    else:
        # CPU only
        max_batch = 16
        batch_size = 8
        min_batch = 4

    return {
        'MAX_CONCURRENT_DOCS': max_concurrent,
        'BATCH_SIZE': batch_size,
        'MIN_BATCH_SIZE': min_batch,
        'MAX_BATCH_SIZE': max_batch,
        'MEMORY_BUFFER': 0.2
    }

@contextmanager
def silent_tqdm():
    """Temporarily disable all tqdm progress bars"""
    original_init = tqdm.__init__

    def silent_init(self, *args, **kwargs):
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)

    tqdm.__init__ = silent_init
    try:
        yield
    finally:
        tqdm.__init__ = original_init


def filter_technical_images(images_data, model, source_doc):
    """
    Filter technical images using CLIP model.
    Identifies technical drawings, diagrams, schematics, and device images
    while filtering out logos, banners, and decorative elements.
    """
    filtered_images = []
    logging.info(f"Processing {len(images_data)} images for technical content")

    # Define classification prompts - ensure all have the same length
    technical_text = "this is a technical diagram or schematic drawing or engineering blueprint"
    non_technical_text = "this is a logo or banner or marketing image or decorative element"

    try:
        # Pre-encode the classification text prompts
        with torch.no_grad():
            with silent_tqdm():
                # Encode both text prompts as single strings
                tech_embedding = model.encode_text([technical_text])
                non_tech_embedding = model.encode_text([non_technical_text])

            if isinstance(tech_embedding, torch.Tensor):
                tech_embedding = tech_embedding.cpu().numpy()
            if isinstance(non_tech_embedding, torch.Tensor):
                non_tech_embedding = non_tech_embedding.cpu().numpy()
    except Exception as err:
        logging.error(f"Error encoding classification prompts: {err}")
        return filtered_images

    for img_data in images_data:
        try:
            image = img_data['image']
            logging.info(f"Image size: {image.size}, mode: {image.mode}")

            # Skip small images that are likely icons or logos
            if image.width < CONFIG.MIN_IMAGE_SIZE or image.height < CONFIG.MIN_IMAGE_SIZE:
                logging.debug(f"Skipping small image: {image.width}x{image.height}")
                continue

            # Skip images with extreme aspect ratios
            aspect_ratio = image.width / image.height
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                logging.debug(f"Skipping image with extreme aspect ratio: {aspect_ratio:.2f}")
                continue

            # Resize image if it's too large
            image = ImageProcessor.resize_image(image, max_size=(CONFIG.MAX_IMAGE_SIZE, CONFIG.MAX_IMAGE_SIZE))

            # Validate and convert image
            if not ImageProcessor.validate_image_data(image):
                continue

            try:
                image = ImageProcessor.convert_to_rgb(image)
            except Exception as err:
                logging.debug(f"Error converting image format: {err}")
                continue

            # Classify image
            try:
                with torch.no_grad():
                    # Encode image
                    with silent_tqdm():
                        image_embedding = model.encode_image([image])
                    if isinstance(image_embedding, torch.Tensor):
                        image_embedding = image_embedding.cpu().numpy()

                    # Calculate similarities using single embeddings
                    tech_similarity = np.dot(image_embedding.flatten(), tech_embedding.flatten())
                    non_tech_similarity = np.dot(image_embedding.flatten(), non_tech_embedding.flatten())

                    # Calculate confidence score
                    total = tech_similarity + non_tech_similarity
                    if total > 0:
                        tech_confidence = tech_similarity / total
                    else:
                        tech_confidence = 0.5

                    if tech_confidence > CONFIG.TECHNICAL_CONFIDENCE_THRESHOLD:
                        img_data['technical_similarity'] = float(tech_confidence)
                        img_data['technical_score'] = float(tech_similarity)
                        filtered_images.append(img_data)
                        logging.info(
                            f"Technical image found in {source_doc} "
                            f"(confidence: {tech_confidence:.4f}, "
                            f"size: {image.width}x{image.height})"
                        )
                    else:
                        logging.debug(
                            f"Non-technical image filtered out: "
                            f"confidence {tech_confidence:.4f}, "
                            f"size: {image.width}x{image.height}"
                        )

            except Exception as err:
                logging.debug(f"Classification error: {str(err)}")
                continue

        except Exception as err:
            logging.debug(f"Error processing image from {source_doc}: {str(err)}")
            continue

    logging.info(f"Found {len(filtered_images)} technical images out of {len(images_data)} total images")
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
        except Exception as err:
            logging.warning(f"Error processing metadata entry {idx}: {err}")
            continue

    # Rebuild index with only valid entries
    try:
        if index is not None and valid_indices:
            new_index = faiss.IndexFlatL2(index.d)
            vectors = np.vstack([index.reconstruct(idx) for idx in valid_indices])
            new_index.add(vectors)
            return cleaned_metadata, new_index
    except Exception as err:
        logging.error(f"Error rebuilding index: {err}")
        return cleaned_metadata, index

    return cleaned_metadata, index


def process_documents(
        model: Any,
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
            except Exception as err:
                logging.warning(f"Could not load existing metadata: {err}")
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
                            with silent_tqdm():
                                text_embeddings, _ = encode_with_clip(chunk_texts, [], model, device)

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
                        images_data = [
                            img for img in images_data
                            if
                            img['image'].width >= CONFIG.MIN_IMAGE_SIZE and img['image'].height >= CONFIG.MIN_IMAGE_SIZE
                        ]
                        filtered_images = filter_technical_images(
                            images_data=images_data,
                            model=model,
                            source_doc=str(doc_path)
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
                                    with silent_tqdm():
                                        text_embeddings, image_embeddings = encode_with_clip(
                                            [], [img_data['image']],
                                            model, device
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

                            except Exception as err:
                                logging.error(f"Error processing image from {doc_path}: {str(err)}")
                                continue

                    pbar.update(1)

                except Exception as err:
                    logging.error(f"Error processing document {doc_path}: {str(err)}")
                    continue

        # Save index and metadata if any embeddings were added
        if embeddings_added:
            save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(metadata, CONFIG.METADATA_PATH)
            logging.info(f"Saved updated FAISS index with {len(metadata)} entries")

        update_processed_files(doc_paths)
        return index, metadata

    except Exception as err:
        logging.error(f"Error during document processing: {str(err)}")
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
                if m.get('type') == 'image' and isinstance(m.get('content'), dict) and m['content'].get('image_id')
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

    except Exception as err:
        logging.error(f"Error updating processed files list: {err}")
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
    except Exception as err:
        logging.error(f"Error getting unprocessed documents: {err}")
        return []


def get_processed_files():
    """Load the list of documents that have already been processed."""
    processed_files_path = CONFIG.BASE_DIR / "processed_files.json"  # Changed to base dir
    try:
        if processed_files_path.exists():
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    except Exception as err:
        logging.error(f"Error loading processed files list: {err}")
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


def init_CLIP_model():
    # Initialize CLIP
    try:
        clip_model, device = CLIP_init(CONFIG.CLIP_MODEL_NAME)

        if clip_model is None:
            raise RuntimeError("Model initialization returned None")

        return clip_model, device
    except Exception as err:
        logging.error(f"CLIP initialization error: {err}", exc_info=True)
        raise RuntimeError(f"Failed to initialize CLIP model: {str(err)}")


def init_FAISS_model():
    # Initialize FAISS
    try:
        CONFIG.RAG_DATA.mkdir(parents=True, exist_ok=True)
        index = None
        metadata = []
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
            except Exception as err:
                logging.warning(f"Failed to load existing index: {err}")
                index = None

        if index is None:
            logging.info(f"Creating new FAISS index with dimension {CONFIG.EMBEDDING_DIMENSION}")
            index = initialize_faiss_index(CONFIG.EMBEDDING_DIMENSION, CONFIG.USE_GPU)
            metadata = []
            save_faiss_index(index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(metadata, CONFIG.METADATA_PATH)

        return index, metadata

    except Exception as err:
        logging.error(f"FAISS initialization error: {err}", exc_info=True)
        raise


def document_processing_sequence(clip_model=None, index=None, metadata=None):
    """Main function with progress bars for batch processing and optimized memory management."""
    if metadata is None:
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

        # Initialize CLIP model
        if clip_model is None:
            clip_model, device = init_CLIP_model()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize FAISS index
        if index is None or metadata is None:
            index, metadata = init_FAISS_model()

        # Process documents in batches
        batch_size = get_optimal_processing_params()['BATCH_SIZE']
        num_batches = (len(new_docs) + batch_size - 1) // batch_size
        with ImageStore() as image_store:
            with tqdm(total=num_batches, desc="Processing batches", unit="batch", position=0, leave=True) as batch_pbar:
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

                    except Exception as err:
                        logging.error(f"Error processing batch {current_batch}: {err}", exc_info=True)
                        continue
                    finally:
                        batch_pbar.update(1)
                        gc.collect()

        logging.info("Processing completed successfully")
        return 0

    except Exception as err:
        logging.error(f"Critical error during processing: {err}", exc_info=True)
        return 1

    finally:
        # Cleanup resources
        try:
            cleanup_and_reload_resources()
            # Continue with the reloaded index and metadata
        except Exception as err:
            logging.error(f"Failed to cleanup and reload resources: {err}")


if __name__ == "__main__":
    setup_logger()

    # Show optimal parameters for your system
    optimal_params = get_optimal_processing_params()
    print("Recommended parameters for your system:")
    for param, value in optimal_params.items():
        print(f"{param} = {value}")

    try:
        result = document_processing_sequence()
        sys.exit(result)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
