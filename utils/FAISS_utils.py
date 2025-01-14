import gc
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Union, Any, Set
from utils.LLM_utils import encode_with_clip
import logging
import numpy as np
import traceback
import faiss
import torch

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
        # Normalize path separators to system-specific ones
        chunk_path = str(chunk_path).replace('\\', '/')
        chunk_path = Path(chunk_path)

        # Construct full path using STORED_TEXT_CHUNKS_PATH
        full_path = CONFIG.STORED_TEXT_CHUNKS_PATH / chunk_path

        # Resolve the path to handle any '..' or '.' components
        full_path = full_path.resolve()

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
                "path": str(relative_path).replace('\\', '/'),
                "type": content_type,
                "chunk": str(Path(relative_path.stem) / chunk_filename).replace('\\', '/'),
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
    Query FAISS with separated image and text search logic.
    First stage: Gets 1% closest text chunks (max 100) and 5% closest images (max 12)
    Second stage: Filters images based on their similarity to the retrieved text content
    """
    try:
        if not metadata:
            logging.info("No documents indexed yet")
            return []

        # 1. Get indices for text chunks and images
        text_indices = [i for i, m in enumerate(metadata) if m.get('type') == 'text-chunk']
        image_indices = [i for i, m in enumerate(metadata) if m.get('type') == 'image']

        if not text_indices and not image_indices:
            logging.info("No content found in metadata")
            return []

        # Calculate quantile limits
        text_limit = min(100, max(1, int(len(text_indices) * 0.01)))  # 1% of texts, max 100
        image_limit = min(12, max(1, int(len(image_indices) * 0.05)))  # 5% of images, max 12

        logging.info(f"Found {len(image_indices)} images and {len(text_indices)} text chunks in metadata")
        logging.info(f"Will return up to {text_limit} text chunks (1%) and {image_limit} images (5%)")

        # 2. Generate query embedding
        query_embedding = None
        if text_query:
            text_embeddings, _ = encode_with_clip(texts=[text_query], images=None, model=model, device=device)
            if text_embeddings is not None:
                query_embedding = text_embeddings
        elif image_query:
            if image_query.mode != "RGB":
                image_query = image_query.convert("RGB")
            _, image_embeddings = encode_with_clip(texts=None, images=[image_query], model=model, device=device)
            if image_embeddings is not None:
                query_embedding = image_embeddings

        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")

        # 3. Perform search with FAISS
        k = len(metadata)  # Get all results initially
        distances, indices = index.search(query_embedding, k)
        similarities = 1 - (np.square(distances[0]) / 2)  # Correct cosine similarity calculation

        # 4. Separate and process results
        text_results = []
        image_results = []

        for idx, sim in zip(indices[0], similarities):
            if idx >= len(metadata):
                continue

            result = {
                "idx": int(idx),
                "metadata": metadata[idx],
                "distance": float(distances[0][idx]),
                "similarity": float(sim)
            }

            if idx in text_indices:
                text_results.append(result)
            elif idx in image_indices:
                image_results.append(result)

        # 5. Sort results by similarity
        text_results.sort(key=lambda x: x['similarity'], reverse=True)
        image_results.sort(key=lambda x: x['similarity'], reverse=True)

        # 6. Take top results by quantile
        final_text_results = text_results[:text_limit]

        # 7. Generate embedding from text results for second-stage image filtering
        if final_text_results:
            # Collect text content
            text_contents = []
            for result in final_text_results:
                if 'metadata' in result and 'chunk' in result['metadata']:
                    content = get_chunk_text(result['metadata']['chunk'])
                    if content:
                        text_contents.append(content)

            if text_contents:
                # Create context embedding from text results
                context_text = " ".join(text_contents)
                context_embedding, _ = encode_with_clip(texts=[context_text], images=None, model=model, device=device)

                if context_embedding is not None:
                    # Search images using the context embedding
                    img_distances, img_indices = index.search(context_embedding, k)
                    img_similarities = 1 - (np.square(img_distances[0]) / 2)

                    # Filter images based on similarity to text context
                    refined_image_results = []
                    for idx, sim in zip(img_indices[0], img_similarities):
                        if idx >= len(metadata) or idx not in image_indices:
                            continue

                        result = {
                            "idx": int(idx),
                            "metadata": metadata[idx],
                            "distance": float(img_distances[0][idx]),
                            "similarity": float(sim)
                        }
                        refined_image_results.append(result)

                    refined_image_results.sort(key=lambda x: x['similarity'], reverse=True)
                    final_image_results = refined_image_results[:image_limit]
                else:
                    final_image_results = image_results[:image_limit]
            else:
                final_image_results = image_results[:image_limit]
        else:
            final_image_results = image_results[:image_limit]

        # 8. Process results for return
        processed_results = []

        # Process text results
        for result in final_text_results:
            processed_result = {
                "idx": result["idx"],
                "distance": result["distance"],
                "similarity": result["similarity"],
                "metadata": result["metadata"].copy()
            }

            chunk_path = result['metadata'].get('chunk')
            if chunk_path:
                processed_result['metadata']['get_content'] = lambda p=chunk_path: get_chunk_text(p)

            processed_results.append(processed_result)

        # Process image results
        for result in final_image_results:
            processed_result = {
                "idx": result["idx"],
                "distance": result["distance"],
                "similarity": result["similarity"],
                "metadata": result["metadata"].copy()
            }
            processed_results.append(processed_result)

        # Log the results
        text_similarities = [r['similarity'] for r in final_text_results]
        image_similarities = [r['similarity'] for r in final_image_results]

        avg_text_sim = sum(text_similarities) / len(text_similarities) if text_similarities else 0
        avg_image_sim = sum(image_similarities) / len(image_similarities) if image_similarities else 0

        logging.info(
            f"Returning {len(final_text_results)} text results (avg sim: {avg_text_sim:.3f}) and "
            f"{len(final_image_results)} image results (avg sim: {avg_image_sim:.3f})"
        )

        return [processed_results]

    except Exception as e:
        logging.error(f"Error in query_with_context: {str(e)}")
        logging.error(traceback.format_exc())
        return []
    finally:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
