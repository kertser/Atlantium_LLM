import faiss
import json
import numpy as np
import logging
from config import CONFIG

def initialize_faiss_index(dimension, use_gpu=False):
    """
    Initialize a FAISS index for storing embeddings.
    :param dimension: Dimension of the embeddings.
    :param use_gpu: Whether to use GPU acceleration (if available).
    :return: FAISS index instance.
    """
    if dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer.")

    index = faiss.IndexFlatL2(dimension)
    if use_gpu and faiss.get_num_gpus() > 0:
        gpu_resource = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        print("FAISS index initialized on GPU.")
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
            "pdf": pdf_name,
            "type": content_type
        }

        # For text chunks, add content directly
        if content_type == "text-chunk":
            meta_entry["content"] = content
        # For images, spread image metadata
        elif content_type == "image" and isinstance(content, dict):
            meta_entry.update({
                "image_id": content["image_id"],
                "caption": content["caption"],
                "context": content["context"],
                "source_doc": content["source_doc"],
                "page": content.get("page")
            })

        metadata.append(meta_entry)
        logging.info(f"Added {content_type} embedding to FAISS from {pdf_name}")

    except Exception as e:
        logging.error(f"Error adding embedding to FAISS: {e}")
        raise


def query_faiss(index, metadata, query_embeddings, top_k):
    """Query the FAISS index with improved results handling"""
    distances, indices = index.search(query_embeddings, top_k)
    results = []

    for i, idx_list in enumerate(indices):
        query_results = []
        for j, idx in enumerate(idx_list):
            if idx < len(metadata):  # Ensure index is within bounds
                query_results.append({
                    "idx": idx,
                    "metadata": metadata[idx],
                    "distance": float(distances[i][j])  # Convert to float for JSON serialization
                })
        results.append(query_results)

    return results


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

def save_metadata(metadata, filepath):
    """
    Save metadata to a file.
    :param metadata: List of metadata associated with FAISS embeddings.
    :param filepath: Path or string to save the metadata file.
    """
    # Convert Path to string if necessary
    filepath_str = str(filepath)
    with open(filepath_str, 'w') as f:
        json.dump(metadata, f)
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

def load_metadata(filepath):
    """
    Load metadata from a file.
    :param filepath: Path or string to the metadata file.
    :return: List of metadata.
    """
    # Convert Path to string if necessary
    filepath_str = str(filepath)
    with open(filepath_str, 'r') as f:
        metadata = json.load(f)
    print(f"Metadata loaded from {filepath_str}.")
    return metadata