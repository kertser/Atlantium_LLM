# Utils Documentation

## Overview

The utils package provides core functionalities for document processing, image handling, vector operations, and system maintenance. This documentation covers the main utility modules and their components.

## FAISS Utils (`FAISS_utils.py`)

### Vector Operations
```python
def initialize_faiss_index(dimension: int, use_gpu: bool = False) -> faiss.Index:
    """Initialize FAISS index with optional GPU support."""

def add_to_faiss(embedding: np.ndarray, source_file_name: str, content_type: str, 
                 content: Dict, index: faiss.Index, metadata: List[Dict]) -> bool:
    """Add embedding to FAISS index."""

def query_faiss(index, metadata, query_embeddings, top_k) -> List[Dict]:
    """Query FAISS index and return results."""

def optimize_faiss_index(index, metadata) -> Tuple[faiss.Index, List[Dict]]:
    """Optimize FAISS index for memory efficiency."""

def save_faiss_index(index, filepath) -> None:
    """Save FAISS index to file."""

def load_faiss_index(filepath) -> faiss.Index:
    """Load FAISS index from file."""
```

## Image Utils (`img_utils.py`)

### Base Image Processing
```python
class ImageProcessor:
    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB format."""

    @staticmethod
    def calculate_hash(image: Union[Image.Image, str, bytes]) -> Optional[str]:
        """Calculate perceptual hash for image."""

    @staticmethod
    def compare_hashes(hash1: str, hash2: str, threshold: float) -> bool:
        """Compare image hashes using weighted similarity."""
```

### Image Storage
```python
class ImageStore(ImageProcessor):
    def store_image(self, image: Image.Image, source_doc: str, page_num: int,
                   caption: Optional[str] = None, context: Optional[str] = None) -> str:
        """Store image and return ID."""

    def get_image(self, image_id: str) -> Tuple[Optional[Image.Image], Optional[Dict]]:
        """Retrieve image and metadata by ID."""

    def get_base64(self, image_id: str) -> Optional[str]:
        """Get base64 encoded image data."""

    def delete_image(self, image_id: str) -> bool:
        """Delete image and its metadata."""

    def deduplicate_images(self) -> None:
        """Remove duplicate images using perceptual hashing."""
```

### Image Classification
```python
class ImageClassifier(ImageProcessor):
    def classify(self, image: Union[Image.Image, str], labels: List[str]) -> Tuple[str, float]:
        """Perform zero-shot classification."""

    def deduplicate(self, images: List[Dict], similarity_threshold: float) -> List[Dict]:
        """Deduplicate image list based on perceptual hash comparison."""
```

## Document Utils (`document_utils.py`)

### Document Management
```python
def rescan_documents(config: CONFIG) -> tuple[bool, str]:
    """Rescan and process new documents."""

def remove_document_from_rag(doc_path: Path) -> tuple[bool, str]:
    """Remove document and its data from RAG system."""

def delete_folder_from_rag(folder_path: Path) -> tuple[bool, str, List[str]]:
    """Delete folder and remove contents from RAG."""

def rename_folder_in_rag(old_path: Path, new_path: Path) -> tuple[bool, str]:
    """Rename folder and update RAG references."""
```

### File Operations
```python
def validate_folder_name(name: str) -> tuple[bool, str]:
    """Validate folder name."""

def create_folder(parent_path: Path, folder_name: str) -> tuple[bool, str]:
    """Create new folder."""

def sanitize_filename(filepath: Path) -> Tuple[Path, bool]:
    """Sanitize filename and extension."""
```

## RAG Utils (`RAG_utils.py`)

### Content Extraction
```python
def extract_text_and_images_from_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """Extract content from PDF files."""

def extract_text_and_images_from_word(doc_path: str) -> Tuple[str, List[Dict]]:
    """Extract content from Word documents."""

def extract_text_and_images_from_excel(excel_path: str) -> Tuple[str, List[Dict]]:
    """Extract content from Excel files."""
```

### Text Processing
```python
def chunk_text(text: str, source_path: str, chunk_size=CONFIG.CHUNK_SIZE) -> List[Dict]:
    """Split text into processable chunks."""

def extract_text_around_image(page, image_bbox, context_range=100) -> str:
    """Extract text context around image location."""
```

## LLM Utils (`LLM_utils.py`)

### Model Integration
```python
def CLIP_init(model_name: str = "openai/clip-vit-base-patch32"):
    """Initialize CLIP model and processor."""

def encode_with_clip(texts, images, model, processor, device):
    """Generate embeddings for text and images."""

def openai_post_request(messages: list, model_name: str, api_key: str,
                       max_tokens: int = None, temperature: float = None) -> Dict:
    """Send request to OpenAI API with retry handling."""
```

## Error Handling

### Common Patterns
1. Rate Limiting and Retries
2. Memory Management
3. File Format Validation
4. Data Validation
5. Error Logging

### Standard Response Format
```python
{
    'success': bool,
    'data': Optional[Any],
    'error': Optional[str],
    'message': str
}
```

## Related Documentation

- [Technical Reference](../docs/technical-reference.md)
- [Models Documentation](../docs/models.md)
- [Frontend Documentation](../docs/frontend.md)
- [Installation Guide](../docs/installation.md)