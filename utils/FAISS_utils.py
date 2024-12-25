import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Any, Set

import faiss
import numpy as np

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

        if len(metadata) >= CONFIG.MAX_METADATA_SIZE:
            raise ValueError("Metadata size limit exceeded")

        if embedding.ndim != 2 or embedding.shape[1] != index.d:
            raise ValueError(f"Embedding shape mismatch. Expected shape: [1, {index.d}]")

        # Add embedding to index first to ensure it's valid
        initial_total = index.ntotal
        index.add(embedding)

        # Verify addition
        if index.ntotal != initial_total + 1:
            raise ValueError(f"Failed to add {content_type} to FAISS index")

        # Process paths
        source_path = Path(source_file_name)
        if source_path.is_absolute():
            try:
                # Get path relative to RAW_DOCUMENTS_PATH
                relative_path = source_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH)
            except ValueError:
                # If path is not relative to RAW_DOCUMENTS_PATH, use the filename only
                relative_path = source_path.name
        else:
            # Already a relative path, ensure it's clean
            relative_path = Path(str(source_path).replace('Raw Documents\\Raw Documents', 'Raw Documents'))

        meta_entry = {
            "path": str(relative_path),
            "type": content_type
        }

        # Handle images
        if content_type == "image":
            image_id = content.get("image_id")
            if not image_id:
                logging.error("Missing image_id in content")
                return

            if processed_ids is not None:
                if image_id in processed_ids:
                    logging.info(f"Skipping duplicate image {image_id}")
                    return
                processed_ids.add(image_id)

            # Ensure source_doc uses clean relative path
            clean_source_doc = str(relative_path)
            meta_entry["image"] = {
                "id": image_id,
                "page": content.get("page", 1),
                "context": content.get("context", ""),
                "caption": content.get("caption", ""),
                "source_doc": clean_source_doc
            }

            logging.info(f"Adding image metadata: {meta_entry}")

        # Handle text chunks
        elif content_type == "text-chunk":
            if not isinstance(content, dict) or 'text' not in content:
                raise ValueError("Text chunk content must be a dictionary with 'text' field")

            # Create directory for document chunks
            doc_chunks_dir = CONFIG.STORED_TEXT_CHUNKS_PATH / relative_path.stem
            doc_chunks_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique chunk filename
            chunk_number = len(list(doc_chunks_dir.glob('chunk_*.txt')))
            chunk_filename = f"chunk_{chunk_number:03d}.txt"
            chunk_path = doc_chunks_dir / chunk_filename

            # Write chunk to file
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(content['text'])

            # Store path relative to STORED_TEXT_CHUNKS_PATH
            meta_entry["chunk"] = str(Path(relative_path.stem) / chunk_filename)

            # Store additional metadata if present
            if content.get('metadata'):
                meta_entry["meta"] = content['metadata']

        # Add metadata entry
        metadata.append(meta_entry)

        # Log successful addition with type-specific details
        if content_type == "image":
            logging.info(f"Successfully added image {image_id} to FAISS index (metadata entry {len(metadata) - 1})")
        else:
            logging.info(f"Successfully added {content_type} from {relative_path} to FAISS index")

        return True

    except Exception as e:
        logging.error(f"Error adding {content_type} to FAISS: {e}")
        raise


def query_faiss(index, metadata, query_embeddings, top_k):
    """Query FAISS index with separate searches for text and images"""
    try:
        if not metadata:
            logging.info("No documents indexed yet")
            return []

        # Get indices for text and images
        text_indices = [i for i, m in enumerate(metadata) if m.get('type') == 'text-chunk']
        image_indices = [i for i, m in enumerate(metadata) if m.get('type') == 'image']

        logging.info(f"Metadata contains {len(image_indices)} images and {len(text_indices)} text chunks")

        # Search in full index
        k = min(len(metadata), top_k)  # Get more results initially
        distances, indices = index.search(query_embeddings, k)

        # Separate results by type
        text_results = []
        image_results = []

        # First pass - collect all results
        seen_indices = set()
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(metadata):
                continue

            result = {
                "idx": int(idx),
                "metadata": metadata[idx],
                "distance": float(distance)
            }

            # Add to appropriate list based on type
            if idx in text_indices:
                text_results.append(result)
                seen_indices.add(idx)
            elif idx in image_indices:
                image_results.append(result)
                seen_indices.add(idx)

        # If we don't have enough image results, do a targeted image search
        if len(image_results) < top_k and image_indices:
            # Create a mask for image indices
            mask = np.zeros(len(metadata), dtype=bool)
            mask[image_indices] = True

            # Search again with mask
            D, I = index.search(query_embeddings, len(image_indices))

            # Add new image results that weren't found before
            for idx, distance in zip(I[0], D[0]):
                if idx in image_indices and idx not in seen_indices:
                    result = {
                        "idx": int(idx),
                        "metadata": metadata[idx],
                        "distance": float(distance)
                    }
                    image_results.append(result)
                    if len(image_results) >= top_k:
                        break

        # Sort results by distance
        text_results.sort(key=lambda x: x['distance'])
        image_results.sort(key=lambda x: x['distance'])

        # Take top_k of each type
        final_results = text_results[:top_k] + image_results[:top_k]
        results = [final_results]  # Maintain expected return format

        logging.info(f"Returning {len(text_results[:top_k])} text and {len(image_results[:top_k])} image results")
        return results

    except Exception as e:
        logging.error(f"Error in query_faiss: {e}")
        return []


def query_with_context(index, metadata, model, processor, device="cpu", text_query=None, image_query=None, top_k=5):
    """Query FAISS with improved context handling"""
    query_embeddings = []

    # Process text query
    if text_query:
        query_input = processor(text=[text_query], return_tensors="pt", padding=True, truncation=True)
        query_input = {k: v.to(device) for k, v in query_input.items()}
        text_embedding = model.get_text_features(**query_input)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        query_embeddings.append(text_embedding.cpu().detach().numpy())

    # Process image query
    if image_query:
        if image_query.mode != "RGB":
            image_query = image_query.convert("RGB")
        image_input = processor(images=image_query, return_tensors="pt").to(device)
        image_embedding = model.get_image_features(**image_input)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        query_embeddings.append(image_embedding.cpu().detach().numpy())

    if not query_embeddings:
        raise ValueError("At least one of text_query or image_query must be provided")

    # Combine and normalize embeddings
    query_embeddings = np.vstack(query_embeddings)

    # Query FAISS
    results = query_faiss(index, metadata, query_embeddings, top_k)

    if not results or not results[0]:
        logging.error("No results retrieved from FAISS index")
        return []

    # Process results with new metadata structure
    processed_results = []
    for result_group in results:
        processed_group = []
        for result in result_group:
            processed_result = {
                "idx": result["idx"],
                "distance": result["distance"],
                "metadata": result["metadata"].copy()
            }

            if result['metadata']['type'] == 'text-chunk':
                chunk_path = result['metadata'].get('chunk')
                if chunk_path:
                    processed_result['metadata']['get_content'] = lambda p=chunk_path: get_chunk_text(p)

            processed_group.append(processed_result)
        processed_results.append(processed_group)

    return processed_results


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
    """Save metadata to a file with proper merging of existing data"""
    filepath_str = str(filepath)
    try:
        # Load existing metadata if it exists
        existing_metadata = []
        if Path(filepath_str).exists():
            try:
                with open(filepath_str, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not load existing metadata from {filepath_str}, starting fresh")

        # Create a set of existing entry identifiers
        existing_ids = {
            entry.get('image', {}).get('id') if entry.get('type') == 'image'
            else entry.get('chunk') if entry.get('type') == 'text-chunk'
            else None
            for entry in existing_metadata
        }

        # Add new entries that don't already exist
        merged_metadata = existing_metadata.copy()
        for entry in metadata:
            entry_id = (
                entry.get('image', {}).get('id') if entry.get('type') == 'image'
                else entry.get('chunk') if entry.get('type') == 'text-chunk'
                else None
            )
            if entry_id not in existing_ids:
                merged_metadata.append(entry)
                existing_ids.add(entry_id)

        # Clean duplicates and save
        cleaned_metadata = clean_duplicate_entries(merged_metadata)
        with open(filepath_str, 'w', encoding='utf-8') as f:
            json.dump(cleaned_metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"Metadata saved to {filepath_str} ({len(cleaned_metadata)} total entries)")

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
