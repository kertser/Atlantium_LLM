# Utils Documentation

## Overview

The utils package provides core functionality for document processing, image handling, embedding generation, and system maintenance.

## Components

### FAISS Utils (`FAISS_utils.py`)

Manages vector storage and retrieval:

```python
def initialize_faiss_index(dimension: int, use_gpu: bool = False) -> faiss.Index
def add_to_faiss(embedding, source_file_name, content_type, content, index, metadata)
def query_with_context(index, metadata, model, processor, query_text=None, image_query=None)
```

### Image Utils (`image_utils.py`)

Handles image processing and analysis:

```python
def zero_shot_classification(image, labels, model, processor, device)
def deduplicate_images(images: List[Dict], max_images: int = 8)
```

### Document Utils (`document_utils.py`)

Manages document operations:

```python
def rescan_documents(config: CONFIG) -> tuple[bool, str]
def remove_document_from_rag(doc_path: Path)
def create_folder(parent_path: Path, folder_name: str)
```

### RAG Utils (`RAG_utils.py`)

Core RAG functionality:

```python
def extract_text_and_images_from_pdf(pdf_path)
def extract_text_and_images_from_word(doc_path)
def extract_text_and_images_from_excel(excel_path)
def chunk_text(text: str, source_path: str)
```

### LLM Utils (`LLM_utils.py`)

LLM integration helpers:

```python
def CLIP_init(model_name="openai/clip-vit-base-patch32")
def encode_with_clip(texts, images, model, processor, device)
def openai_post_request(messages, model_name, max_tokens, temperature, api_key)
```

## Usage Examples

### Processing Documents
```python
from utils.RAG_utils import extract_text_and_images_from_pdf

text, images = extract_text_and_images_from_pdf('document.pdf')
```

### Managing Images
```python
from utils.image_utils import deduplicate_images

unique_images = deduplicate_images(images, max_images=8)
```

### Vector Operations
```python
from utils.FAISS_utils import initialize_faiss_index, add_to_faiss

index = initialize_faiss_index(dimension=512, use_gpu=True)
add_to_faiss(embedding, source_file, 'text-chunk', content, index, metadata)
```

## Best Practices

1. **Document Processing**
   - Validate file formats
   - Handle large files in chunks
   - Maintain proper error handling

2. **Image Management**
   - Check image quality
   - Handle various formats
   - Implement proper caching

3. **Vector Operations**
   - Use appropriate batch sizes
   - Implement proper indexing
   - Handle out-of-memory scenarios

## Error Handling

Common issues and solutions:
1. Memory Management
2. File Format Handling
3. Network Timeouts
4. GPU Availability
