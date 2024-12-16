from typing import List, Dict, Union, Any, Set
from pathlib import Path
import faiss
import json
import numpy as np
import logging
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
    """Remove duplicate image entries from metadata"""
    cleaned_metadata = []
    seen_image_ids = set()

    for entry in metadata:
        if entry['type'] == 'image':
            image_id = entry.get('content', {}).get('image_id', '')
            if image_id and image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
                cleaned_metadata.append(entry)
        else:
            cleaned_metadata.append(entry)

    return cleaned_metadata


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
    """Add embedding to FAISS with optimized metadata storage."""
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

        # Process paths
        source_path = Path(source_file_name)
        if not source_path.is_absolute():
            source_path = CONFIG.RAW_DOCUMENTS_PATH / source_path

        try:
            relative_path = source_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH)
            meta_entry = {
                "path": str(relative_path),
                "type": content_type
            }
        except ValueError as e:
            logging.error(f"Error processing path {source_path}: {e}")
            raise

        # Handle images
        if content_type == "image" and processed_ids is not None:
            image_id = content.get("image_id", "")
            if image_id in processed_ids:
                logging.info(f"Skipping duplicate image {image_id}")
                return
            processed_ids.add(image_id)

        # Add embedding to index
        index.add(embedding)

        # Process content based on type
        if content_type == "text-chunk":
            if isinstance(content, dict):
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

                # Store only non-empty additional metadata
                extra_meta = content.get('metadata', {})
                if extra_meta:
                    meta_entry["meta"] = {k: v for k, v in extra_meta.items() if v}
            else:
                raise ValueError("Text chunk content must be a dictionary")

        elif content_type == "image":
            if isinstance(content, dict):
                # Store only essential image information
                image_data = {
                    "id": content.get("image_id", ""),
                    "page": content.get("page", 1)
                }

                # Only store non-empty fields
                if content.get("context"):
                    image_data["context"] = content["context"]
                if content.get("caption"):
                    image_data["caption"] = content["caption"]

                meta_entry["image"] = image_data

        metadata.append(meta_entry)
        logging.info(f"Added {content_type} embedding from {relative_path}")

    except Exception as e:
        logging.error(f"Error adding embedding to FAISS: {e}")
        raise

def query_faiss(index, metadata, query_embeddings, top_k):
    """Query FAISS index with improved error handling"""
    try:
        # Handle empty metadata case
        if not metadata:
            logging.info("No documents indexed yet")
            return []

        # Add logging
        logging.info(f"Querying FAISS index with {len(metadata)} total entries")

        # Get all results, but ensure k is at least 1
        k = max(1, min(len(metadata), top_k * 2))
        distances, indices = index.search(query_embeddings, k)

        logging.info(f"Query returned {len(indices[0])} results")

        results = []
        text_results = []
        image_results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(metadata):
                continue
            result = {
                "idx": int(idx),
                "metadata": metadata[idx],
                "distance": float(distance)
            }
            if metadata[idx].get('type') == 'image':
                image_results.append(result)
            else:
                text_results.append(result)

        # Combine results with logging
        results.append(text_results[:top_k] + image_results[:top_k])
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
    results = query_faiss(index, metadata, query_embeddings, top_k * 2)

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

def save_faiss_index(index, filepath):
    """Save the FAISS index to a file."""
    filepath_str = str(filepath)
    faiss.write_index(index, filepath_str)
    logging.info(f"FAISS index saved to {filepath_str}")  # Changed to logging


def save_metadata(metadata: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """Save metadata to a file with duplicate cleaning"""
    filepath_str = str(filepath)
    cleaned_metadata = clean_duplicate_entries(metadata)
    with open(filepath_str, 'w', encoding='utf-8') as f:
        json.dump(cleaned_metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"Metadata saved to {filepath_str}")  # Changed to logging

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
