import hashlib
import json
from pathlib import Path
from typing import List, Dict, Union, Any, Set
from utils.LLM_utils import encode_with_clip
import logging
import numpy as np
import traceback
import faiss

from config import CONFIG


# Define GPU functions at module level
def _get_gpu_resources():
    try:
        from faiss.swigfaiss import StandardGpuResources as SGR
        return SGR()
    except ImportError:
        return None


def _cpu_to_gpu(res, dev: int, index):
    try:
        from faiss.swigfaiss import index_cpu_to_gpu as i2g
        return i2g(res, dev, index)
    except ImportError:
        return index


def initialize_faiss_index(dimension: int, use_gpu: bool = False) -> faiss.Index:
    """
    Initialize a FAISS index for storing embeddings.
    Args:
        dimension: Dimension of the embeddings.
        use_gpu: Whether to use GPU acceleration (if available).
    Returns:
        FAISS index instance.
    """
    if dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer.")

    index = faiss.IndexFlatL2(dimension)
    if use_gpu and faiss.get_num_gpus() > 0:
        gpu_resource = _get_gpu_resources()
        if gpu_resource is not None:
            index = _cpu_to_gpu(gpu_resource, 0, index)
            logging.info("GPU support available in FAISS installation.")
        else:
            logging.warning("GPU support not available in FAISS installation.")
    else:
        logging.info("Using CPU for FAISS index.")
    return index


def clean_duplicate_entries(metadata: List[Dict]) -> List[Dict]:
    """Remove duplicate entries from metadata with improved deduplication logic"""
    if not metadata:
        return []

    cleaned_metadata = []
    seen_entries = {}  # Using dict for O(1) lookup

    for entry in metadata:
        try:
            if entry['type'] == 'image':
                # For images, use image_id as unique identifier
                key = ('image', entry.get('image', {}).get('id'))
            elif entry['type'] == 'text-chunk':
                # For text chunks, use combination of path and chunk number
                key = ('text', entry.get('path'), entry.get('chunk'))
            else:
                # For other types, use content hash
                content_str = json.dumps(entry.get('content', {}), sort_keys=True)
                key = ('other', hashlib.md5(content_str.encode()).hexdigest())

            # Keep only the most recent entry for each unique key
            if key not in seen_entries:
                seen_entries[key] = entry
                cleaned_metadata.append(entry)

        except Exception as e:
            logging.warning(f"Error processing metadata entry during cleanup: {e}")
            continue

    logging.info(f"Cleaned metadata: removed {len(metadata) - len(cleaned_metadata)} duplicate entries")
    return cleaned_metadata


def validate_metadata_integrity(metadata: List[Dict]) -> List[Dict]:
    """Validate metadata entries and remove invalid ones"""
    valid_metadata = []

    for entry in metadata:
        try:
            # Check required fields
            if not isinstance(entry, dict):
                continue

            if 'type' not in entry or 'path' not in entry:
                continue

            # Type-specific validation
            if entry['type'] == 'image':
                if not entry.get('image', {}).get('id'):
                    continue
            elif entry['type'] == 'text-chunk':
                if not entry.get('chunk'):
                    continue

            valid_metadata.append(entry)

        except Exception as e:
            logging.warning(f"Invalid metadata entry: {e}")
            continue

    if len(valid_metadata) < len(metadata):
        logging.warning(f"Removed {len(metadata) - len(valid_metadata)} invalid metadata entries")

    return valid_metadata


def get_chunk_text(chunk_path: str) -> str:
    """Retrieve text chunk content from file."""
    try:
        # Construct full path using STORED_TEXT_CHUNKS_PATH
        full_path = CONFIG.STORED_TEXT_CHUNKS_PATH / chunk_path
        if not full_path.exists():
            logging.error(f"Chunk file not found: {full_path}")
            return ""

        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error reading chunk file {chunk_path}: {e}")
        return ""


def add_to_faiss(embedding, source_file_name, content_type, content, index, metadata, processed_ids: Set[str] = None):
    """
    Add embedding to FAISS with optimized metadata storage and improved error handling.

    Args:
        embedding: NumPy array of the embedding
        source_file_name: Path to the source file
        content_type: Type of content ('text-chunk' or 'image')
        content: Dictionary containing content details
        index: FAISS index instance
        metadata: List of metadata entries
        processed_ids: Set of already processed image IDs (for deduplication)

    Returns:
        None
    """
    try:
        # Validate embedding
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a valid NumPy array")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if embedding.ndim != 2 or embedding.shape[1] != index.d:
            raise ValueError(f"Embedding shape mismatch. Expected shape: [1, {index.d}]")

        # Process paths
        source_path = Path(source_file_name)
        if source_path.is_absolute():
            try:
                relative_path = source_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH)
            except ValueError:
                relative_path = source_path.name
        else:
            relative_path = Path(str(source_path).replace('Raw Documents\\Raw Documents', 'Raw Documents'))

        # Handle text chunks
        if content_type == "text-chunk":
            if not isinstance(content, dict) or 'text' not in content:
                raise ValueError("Text chunk content must be a dictionary with 'text' field")

            # Generate a hash of the chunk content
            chunk_hash = hashlib.md5(content['text'].encode()).hexdigest()

            # Check if this chunk already exists in metadata
            chunk_exists = False
            for entry in metadata:
                if (entry.get('type') == 'text-chunk' and
                        entry.get('chunk_hash') == chunk_hash and
                        entry.get('path') == str(relative_path)):
                    chunk_exists = True
                    break

            if chunk_exists:
                logging.info(f"Skipping duplicate chunk for {relative_path}")
                return False

            # Create directory for document chunks
            doc_chunks_dir = CONFIG.STORED_TEXT_CHUNKS_PATH / relative_path.stem
            doc_chunks_dir.mkdir(parents=True, exist_ok=True)

            # Find the next available chunk number by checking existing files
            existing_chunks = list(doc_chunks_dir.glob('chunk_*.txt'))
            existing_numbers = {
                int(chunk_file.stem.split('_')[1])
                for chunk_file in existing_chunks
                if chunk_file.stem.split('_')[1].isdigit()
            }

            # Find the first available number
            chunk_number = 0
            while chunk_number in existing_numbers:
                chunk_number += 1

            chunk_filename = f"chunk_{chunk_number:03d}.txt"
            chunk_path = doc_chunks_dir / chunk_filename

            # Add embedding to index first (before writing file)
            initial_total = index.ntotal
            index.add(embedding)

            # Verify addition
            if index.ntotal != initial_total + 1:
                raise ValueError("Failed to add embedding to FAISS index")

            # Write chunk to file only after successful embedding addition
            try:
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(content['text'])
            except Exception as e:
                # If file write fails, we should ideally remove the embedding we just added
                # However, FAISS doesn't provide a direct way to remove the last added embedding
                # So we'll at least log the error
                logging.error(f"Failed to write chunk file {chunk_path}: {e}")
                raise

            # Create metadata entry
            meta_entry = {
                "path": str(relative_path),
                "type": content_type,
                "chunk": str(Path(relative_path.stem) / chunk_filename),
                "chunk_hash": chunk_hash
            }

            if content.get('metadata'):
                meta_entry["meta"] = content['metadata']

            metadata.append(meta_entry)
            logging.info(f"Added text chunk {chunk_number} from {relative_path}")
            return True

        # Handle images
        elif content_type == "image":
            image_id = content.get("image_id")
            if not image_id:
                logging.error("Missing image_id in content")
                return False

            if processed_ids is not None and image_id in processed_ids:
                logging.info(f"Skipping duplicate image {image_id}")
                return False

            # Add embedding to index
            initial_total = index.ntotal
            index.add(embedding)

            if index.ntotal != initial_total + 1:
                raise ValueError("Failed to add image embedding to FAISS index")

            meta_entry = {
                "path": str(relative_path),
                "type": content_type,
                "image": {
                    "id": image_id,
                    "page": content.get("page", 1),
                    "context": content.get("context", ""),
                    "caption": content.get("caption", ""),
                    "source_doc": str(relative_path)
                }
            }

            if processed_ids is not None:
                processed_ids.add(image_id)

            metadata.append(meta_entry)
            logging.info(f"Added image {image_id} to FAISS index")
            return True

        return False

    except Exception as e:
        logging.error(f"Error adding {content_type} to FAISS: {e}")
        raise

def query_with_context(index, metadata, model, device="cpu", text_query=None, image_query=None):
    """
    Query FAISS with adaptive thresholding and dynamic top_k based on corpus size.
    """
    try:
        if not metadata:
            logging.info("No documents indexed yet")
            return []

        query_embeddings = []
        is_text_query = text_query is not None
        is_image_query = image_query is not None

        # Process text query
        if text_query:
            text_embeddings, _ = encode_with_clip(texts=[text_query], images=None, model=model, device=device)
            if text_embeddings is not None:
                query_embeddings.append(text_embeddings)

        # Process image query
        if image_query:
            if image_query.mode != "RGB":
                image_query = image_query.convert("RGB")
            _, image_embeddings = encode_with_clip(texts=None, images=[image_query], model=model, device=device)
            if image_embeddings is not None:
                query_embeddings.append(image_embeddings)

        if not query_embeddings:
            raise ValueError("At least one of text_query or image_query must be provided")

        query_embeddings = np.vstack(query_embeddings)

        # Get indices for text and images
        text_indices = [i for i, m in enumerate(metadata) if m.get('type') == 'text-chunk']
        image_indices = set(i for i, m in enumerate(metadata) if m.get('type') == 'image')

        logging.info(f"Found {len(image_indices)} images and {len(text_indices)} text chunks in metadata")

        def calculate_adaptive_k(total_items, base_percentage=0.05, min_k=2, max_k=30):
            if total_items == 0:
                return 0
            top_k = max(min_k, min(max_k, int(total_items * base_percentage)))
            if total_items < 20:
                top_k = min(top_k, total_items)
            return top_k

        text_top_k = calculate_adaptive_k(
            len(text_indices),
            base_percentage=0.08,
            min_k=3,
            max_k=30
        )

        image_top_k = calculate_adaptive_k(
            len(image_indices),
            base_percentage=0.05,
            min_k=2,
            max_k=20
        )

        logging.info(f"Adaptive top_k values - Text: {text_top_k}, Image: {image_top_k}")

        k = min(len(metadata), max(text_top_k + image_top_k, 50))
        distances, indices = index.search(query_embeddings, k)
        similarities = 1 - (distances[0] / 2)

        text_similarities = []
        image_similarities = []

        for idx, sim in zip(indices[0], similarities):
            if idx >= len(metadata):
                continue
            if idx in text_indices:
                text_similarities.append(sim)
            elif idx in image_indices:
                image_similarities.append(sim)

        def calculate_adaptive_threshold(sims, content_type):
            if len(sims) == 0:
                return 0.3 if content_type == 'text' else 0.05

            mean = np.mean(sims)
            std = np.std(sims)
            q75, q25 = np.percentile(sims, [75, 25])
            iqr = q75 - q25

            if content_type == 'text':
                base_multiplier = 0.6
                min_threshold = 0.3
            else:
                base_multiplier = 0.15
                min_threshold = 0.05

            if std > 0.1:
                threshold = mean + base_multiplier * std
            else:
                threshold = q75 - base_multiplier * iqr

            if content_type == 'image':
                threshold = max(min(threshold, mean * 0.8), min_threshold)
            else:
                threshold = max(min(threshold, mean), q25, min_threshold)

            return threshold

        text_threshold = calculate_adaptive_threshold(text_similarities, 'text')
        image_threshold = calculate_adaptive_threshold(image_similarities, 'image')

        if is_text_query and not is_image_query:
            image_threshold *= 1.2
        elif is_image_query and not is_text_query:
            image_threshold *= 0.5
        else:
            image_threshold *= 0.8

        logging.info(f"Final thresholds - Text: {text_threshold:.3f}, Image: {image_threshold:.3f}")

        text_results = []
        image_results = []
        seen_indices = set()

        for idx, sim in zip(indices[0], similarities):
            if idx >= len(metadata):
                continue

            result = {
                "idx": int(idx),
                "metadata": metadata[idx],
                "distance": float(1 - sim),
                "similarity": float(sim)
            }

            if idx in text_indices and sim >= text_threshold:
                text_results.append(result)
                seen_indices.add(idx)
            elif idx in image_indices and sim >= image_threshold:
                image_results.append(result)
                seen_indices.add(idx)

        if len(image_results) < image_top_k and image_indices:
            extra_k = min(len(metadata), 200)
            image_distances, image_idx = index.search(query_embeddings, extra_k)

            lenient_threshold = max(0.03, image_threshold * 0.3)

            for idx, distance in zip(image_idx[0], image_distances[0]):
                if idx in image_indices and idx not in seen_indices:
                    sim = 1 - (distance / 2)
                    if sim >= lenient_threshold:
                        result = {
                            "idx": int(idx),
                            "metadata": metadata[idx],
                            "distance": float(distance),
                            "similarity": float(sim)
                        }
                        image_results.append(result)
                        seen_indices.add(idx)

                        if len(image_results) >= image_top_k:
                            break

        text_results.sort(key=lambda x: x['similarity'], reverse=True)
        image_results.sort(key=lambda x: x['similarity'], reverse=True)

        final_results = text_results[:text_top_k] + image_results[:image_top_k]

        processed_results = []
        for result in final_results:
            processed_result = {
                "idx": result["idx"],
                "distance": result["distance"],
                "similarity": result["similarity"],
                "metadata": result["metadata"].copy()
            }

            if result['metadata']['type'] == 'text-chunk':
                chunk_path = result['metadata'].get('chunk')
                if chunk_path:
                    processed_result['metadata']['get_content'] = lambda p=chunk_path: get_chunk_text(p)

            processed_results.append(processed_result)

        logging.info(
            f"Returning {len(text_results[:text_top_k])} text results (threshold: {text_threshold:.3f}) and "
            f"{len(image_results[:image_top_k])} image results (threshold: {image_threshold:.3f})"
        )

        return [processed_results]

    except Exception as e:
        logging.error(f"Error in query_with_context: {str(e)}")
        logging.error(traceback.format_exc())
        return []

def optimize_faiss_index(index, metadata):
    """Optimize FAISS index for memory efficiency"""
    if not metadata or not index:
        return index, metadata

    try:
        # Remove duplicate vectors
        unique_vectors = {}
        for idx, meta in enumerate(metadata):
            try:
                vector = index.reconstruct(idx)
                vector_hash = hashlib.md5(vector.tobytes()).hexdigest()
                if vector_hash not in unique_vectors:
                    unique_vectors[vector_hash] = (vector, meta)
            except Exception as e:
                logging.warning(f"Error processing vector {idx}: {e}")
                continue

        # Rebuild index with unique vectors
        new_index = faiss.IndexFlatL2(index.d)
        new_metadata = []

        for _, (vector, meta) in unique_vectors.items():
            try:
                new_index.add(np.array([vector]))
                new_metadata.append(meta)
            except Exception as e:
                logging.warning(f"Error adding vector to new index: {e}")
                continue

        return new_index, new_metadata

    except Exception as e:
        logging.error(f"Error optimizing index: {e}")
        return index, metadata


def save_faiss_index(index, filepath):
    """Save the FAISS index to a file."""
    filepath_str = str(filepath)
    faiss.write_index(index, filepath_str)
    logging.info(f"FAISS index saved to {filepath_str}")  # Changed to logging


def save_metadata(metadata: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """Save metadata to a file"""
    filepath_str = str(filepath)
    try:
        # Write the metadata directly to file
        with open(filepath_str, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"Metadata saved to {filepath_str} ({len(metadata)} entries)")
    except Exception as e:
        logging.error(f"Error saving metadata to {filepath_str}: {e}")
        raise


def load_faiss_index(filepath):
    """
    Load the FAISS index from a file.
    :param filepath: Path or string to the FAISS index file.
    :return: Loaded FAISS index.
    """
    # Convert Path to string if necessary
    filepath_str = str(filepath)
    index = faiss.read_index(filepath_str)
    logging.info(f"FAISS index loaded from {filepath_str}")
    return index


def load_metadata(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load metadata from a file and ensure no duplicates"""
    filepath_str = str(filepath)
    with open(filepath_str, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Clean any existing duplicates
    cleaned_metadata = clean_duplicate_entries(metadata)

    # If we cleaned any duplicates, save the cleaned version
    if len(cleaned_metadata) != len(metadata):
        logging.info(f"Removed {len(metadata) - len(cleaned_metadata)} duplicate entries")
        save_metadata(cleaned_metadata, filepath)

    logging.info(f"Loaded metadata from {filepath_str}")
    return cleaned_metadata
