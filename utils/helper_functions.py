import gc
import torch
from utils.FAISS_utils import load_faiss_index, load_metadata, optimize_faiss_index
from utils.document_utils import cleanup_orphaned_chunks
import logging
from config import CONFIG


def cleanup_and_reload_resources():
    """
    Cleanup system resources and reload models/indices.
    This function performs the following operations:
    1. Cleans up GPU memory if available
    2. Runs garbage collection
    3. Reloads FAISS index and metadata
    4. Optimizes the FAISS index
    5. Cleans up orphaned chunks

    Returns:
        tuple: (index, metadata)
            - index: FAISS index object
            - metadata: List containing metadata for the index
    """

    try:
        # Step 1: Clear CUDA cache if GPU is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Run garbage collection
        gc.collect()

        # Step 3: Reload FAISS index and metadata
        index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
        metadata = load_metadata(CONFIG.METADATA_PATH)
        logging.info("Successfully reloaded FAISS index and metadata")

        # Step 4: Optimize FAISS index
        optimized_index, optimized_metadata = optimize_faiss_index(index, metadata)
        if optimized_index is not None and optimized_metadata is not None:
            index = optimized_index
            metadata = optimized_metadata
            logging.info("Successfully optimized FAISS index")

        # Step 5: Cleanup orphaned chunks
        cleanup_success, cleanup_msg = cleanup_orphaned_chunks()
        if not cleanup_success:
            logging.warning(f"Chunk cleanup warning: {cleanup_msg}")

        return index, metadata

    except Exception as e:
        error_msg = f"Error during cleanup and reload: {str(e)}"
        logging.error(error_msg, exc_info=True)
        raise  # Re-raise the exception to be handled by the caller
