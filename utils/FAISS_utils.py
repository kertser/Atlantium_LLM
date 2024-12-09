from typing import List, Dict, Union, Any
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
            print("FAISS index initialized on GPU.")
        else:
            print("GPU support not available in FAISS installation.")
    else:
        print("FAISS index initialized on CPU.")
    return index


def add_to_faiss(embedding, pdf_name, content_type, content, index, metadata):
    """Add embedding to FAISS with improved metadata handling"""
    try:
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a valid NumPy array")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if len(metadata) >= CONFIG.MAX_METADATA_SIZE:
            raise ValueError("Metadata size limit exceeded")

        if embedding.ndim != 2 or embedding.shape[1] != index.d:
            raise ValueError(f"Embedding shape mismatch. Expected shape: [1, {index.d}]")

        index.add(embedding)

        # Create base metadata
        meta_entry = {
            "pdf": str(pdf_name),  # Convert Path to string if needed
            "type": content_type
        }

        # For text chunks, add content directly
        if content_type == "text-chunk":
            meta_entry["content"] = content

        # For images, handle potentially missing fields with defaults
        elif content_type == "image":
            if isinstance(content, dict):
                doc_name = Path(content.get("source_doc", pdf_name)).name
                meta_entry.update({
                    "image_id": content.get("image_id", ""),
                    "caption": content.get("caption", f"Image from {doc_name}"),
                    "context": content.get("context", ""),
                    "source_doc": str(content.get("source_doc", pdf_name)),
                    "page": content.get("page", 1)
                })
            else:
                doc_name = Path(pdf_name).name
                meta_entry.update({
                    "image_id": "",
                    "caption": f"Image from {doc_name}",
                    "context": "",
                    "source_doc": str(pdf_name),
                    "page": 1
                })

        metadata.append(meta_entry)
        logging.info(f"Added {content_type} embedding to FAISS from {pdf_name}")

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
    results = query_faiss(index, metadata, query_embeddings, top_k * 2)  # Get more results for filtering

    if not results or not results[0]:
        logging.error("No results retrieved from FAISS index")
        return []

    return results

def save_faiss_index(index, filepath):
    """
    Save the FAISS index to a file.
    :param index: FAISS index to save.
    :param filepath: Path or string to save the FAISS index file.
    """
    # Convert Path to string if necessary
    filepath_str = str(filepath)
    faiss.write_index(index, filepath_str)
    print(f"FAISS index saved to {filepath_str}.")

def save_metadata(metadata: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """
    Save metadata to a file.
    Args:
        metadata: List of metadata associated with FAISS embeddings.
        filepath: Path or string to save the metadata file.
    """
    filepath_str = str(filepath)
    with open(filepath_str, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {filepath_str}.")

def load_faiss_index(filepath):
    """
    Load the FAISS index from a file.
    :param filepath: Path or string to the FAISS index file.
    :return: Loaded FAISS index.
    """
    # Convert Path to string if necessary
    filepath_str = str(filepath)
    index = faiss.read_index(filepath_str)
    print(f"FAISS index loaded from {filepath_str}.")
    return index

def load_metadata(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load metadata from a file.
    Args:
        filepath: Path or string to the metadata file.
    Returns:
        List of metadata dictionaries.
    """
    filepath_str = str(filepath)
    with open(filepath_str, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Metadata loaded from {filepath_str}.")
    return metadata
