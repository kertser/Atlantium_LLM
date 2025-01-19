# Utils Documentation

## Overview

The utils package provides core functionalities for document processing, image handling, vector operations, and system utilities for the Atlantium LLM system. Each utility module is designed for specific tasks in the RAG-based technical assistant.

## FAISS Utils (`FAISS_utils.py`)

### Vector Operations
- **initialize_faiss_index**: Creates FAISS index with optional GPU support.
- **add_to_faiss**: Adds embeddings to FAISS index with metadata.
- **query_faiss**: Performs similarity search on FAISS index.
- **optimize_faiss_index**: Optimizes index for memory efficiency.
- **save_faiss_index**: Persists index to storage.
- **load_faiss_index**: Loads index from storage.

## Image Utils (`img_utils.py`)

### Base Image Processing
- **ImageProcessor Class**:
  - `convert_to_rgb`: Converts images to RGB format.
  - `calculate_hash`: Generates perceptual hash for images.
  - `compare_hashes`: Compares image hashes with similarity threshold.

### Image Storage
- **ImageStore Class**:
  - `store_image`: Saves images with metadata and context.
  - `get_image`: Retrieves images and associated metadata.
  - `get_base64`: Gets base64 encoded image data.
  - `delete_image`: Removes images from storage.
  - `deduplicate_images`: Removes duplicate images using perceptual hashing.

### Image Classification
- **ImageClassifier Class**:
  - `classify`: Performs zero-shot classification of technical images.
  - `deduplicate`: Deduplicates images based on perceptual similarity.

## Document Utils (`document_utils.py`)

### Document Management
- **rescan_documents**: Processes new documents in the system.
- **remove_document_from_rag**: Removes documents from RAG system.
- **delete_folder_from_rag**: Deletes folders and their contents.
- **rename_folder_in_rag**: Updates folder references in RAG.

### File Operations
- **validate_folder_name**: Validates folder name format.
- **create_folder**: Creates new folders with proper permissions.
- **sanitize_filename**: Cleans and validates file names.

## RAG Utils (`RAG_utils.py`)

### Content Extraction
- **extract_text_and_images_from_pdf**: Extracts PDF content.
- **extract_text_and_images_from_word**: Processes Word documents.
- **extract_text_and_images_from_excel**: Extracts Excel data.

### Text Processing
- **chunk_text**: Splits text into processable chunks.
- **extract_text_around_image**: Extracts text context near images.

## LLM Utils (`LLM_utils.py`)

### Model Integration
- **CLIP_init**: Initializes CLIP model for image-text processing.
- **encode_with_clip**: Generates text and image embeddings.
- **openai_post_request**: Handles OpenAI API communication.

## Error Handling

### Common Patterns
1. **Rate Limiting and Retries**: Implements retry mechanisms to handle API rate limits.
2. **Memory Management**: Ensures efficient memory use during processing.
3. **File Format Validation**: Verifies the compatibility of files.
4. **Data Validation**: Ensures the accuracy and integrity of inputs.
5. **Error Logging**: Records issues for debugging.

### Standard Response Format

Responses from utility methods follow this structure:
```json
{
    "success": boolean,
    "data": Optional[Any],
    "error": Optional[string],
    "message": string
}
```

## Related Documentation

- [Technical Reference](../docs/technical-reference.md)
- [Models Documentation](../docs/models.md)
- [Frontend Documentation](../docs/frontend.md)
- [Installation Guide](../docs/installation.md)

