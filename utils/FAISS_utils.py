import faiss
import json
import numpy as np
import logging

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
    """
    Add a single embedding to the FAISS index with associated metadata.
    :param embedding: Numpy array of the embedding vector.
    :param pdf_name: Name of the source document.
    :param content_type: Type of the content (e.g., "text-chunk", "image").
    :param content: The actual content (text or image description).
    """

    try:
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a valid NumPy array.")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.ndim != 2 or embedding.shape[1] != index.d:
            raise ValueError(f"Embedding shape mismatch. Expected shape: [1, {index.d}]")
        index.add(embedding)
        metadata.append({"pdf": pdf_name, "type": content_type, "content": content})
        print(f"Added embedding to FAISS: {pdf_name}, type: {content_type}")
    except Exception as e:
        print(f"Error adding embedding to FAISS: {e}")

def query_faiss(index, metadata, query_embeddings, top_k):
    """
    Query the FAISS index and retrieve the top_k results along with metadata.
    """
    distances, indices = index.search(query_embeddings, top_k)
    results = []
    for i, idx_list in enumerate(indices):
        query_results = []
        for j, idx in enumerate(idx_list):
            if idx < len(metadata):  # Ensure index is within bounds
                query_results.append({
                    "metadata": metadata[idx],
                    "distance": distances[i][j]
                })
        results.append(query_results)
    return results

def query_with_context(index, metadata, model, processor, device="cpu", text_query=None, image_query=None, top_k=5):
    """
    Query FAISS index with text and/or image.
    """
    query_embeddings = []

    # Encode text query
    if text_query:
        query_input = processor(text=[text_query], return_tensors="pt", padding=True, truncation=True)
        query_input = {k: v.to(device) for k, v in query_input.items()}
        text_embedding = model.get_text_features(**query_input)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        query_embeddings.append(text_embedding.cpu().detach().numpy())

    # Encode image query
    if image_query:
        if image_query.mode != "RGB":
            image_query = image_query.convert("RGB")
        image_input = processor(images=image_query, return_tensors="pt").to(device)
        image_embedding = model.get_image_features(**image_input)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        query_embeddings.append(image_embedding.cpu().detach().numpy())

    # Combine query embeddings
    if not query_embeddings:
        raise ValueError("At least one of text_query or image_query must be provided.")
    query_embeddings = np.vstack(query_embeddings)

    # Query FAISS index
    results = query_faiss(index, metadata, query_embeddings, top_k)
    if not results or not results[0]:
        logging.error("No results retrieved from FAISS index.")
    return results

def save_faiss_index(index, filepath):
    """
    Save the FAISS index to a file.
    :param index: FAISS index to save.
    :param filepath: Path to save the FAISS index file.
    """
    faiss.write_index(index, filepath)
    print(f"FAISS index saved to {filepath}.")

def save_metadata(metadata, filepath):
    """
    Save metadata to a file.
    :param metadata: List of metadata associated with FAISS embeddings.
    :param filepath: Path to save the metadata file.
    """
    with open(filepath, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {filepath}.")

def load_faiss_index(filepath):
    """
    Load the FAISS index from a file.
    :param filepath: Path to the FAISS index file.
    :return: Loaded FAISS index.
    """
    index = faiss.read_index(filepath)
    print(f"FAISS index loaded from {filepath}.")
    return index

def load_metadata(filepath):
    """
    Load metadata from a file.
    :param filepath: Path to the metadata file.
    :return: List of metadata.
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    print(f"Metadata loaded from {filepath}.")
    return metadata