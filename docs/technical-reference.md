# Technical Reference Documentation

## System Architecture

```
Atlantium_LLM/
├── server.py                 # Main FastAPI server
├── run.py                   # Server runner
├── RAG_processor.py         # Document processor
├── config.py               # System configuration
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── prompt_loader.py    # Prompt management
│   ├── prompts.py         # Prompt building
│   └── templates/
│       └── prompts.yaml   # Prompt templates
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── FAISS_utils.py     # Vector operations
│   ├── image_utils.py     # Image processing
│   ├── LLM_utils.py       # LLM integration
│   ├── RAG_utils.py       # Document processing
│   ├── document_utils.py  # File management
│   └── image_store.py     # Image storage
└── static/                # Frontend assets
    ├── index.html
    ├── styles.css
    ├── scripts.js
    └── favicon.png
```

## Core Classes and Methods

### Server Components

#### RAGQueryServer (server.py)
```python
class RAGQueryServer:
    def __init__(self)
    def determine_query_type(query_text: str) -> QueryType
    def get_relevant_contexts(results: List[Dict], query_text: str) -> Tuple[List[str], List[Dict]]
    def prepare_prompt(query_text: str, contexts: List[str], query_type: QueryType, images: List[Dict]) -> str
    def process_text_query(query_text: str, top_k: int = CONFIG.DEFAULT_TOP_K) -> QueryResponse
    async def process_image_query(image_data: bytes, query_text: Optional[str] = None) -> str
```

### Model Components

#### PromptLoader (models/prompt_loader.py)
```python
class PromptLoader:
    def __init__(self)
    def get_system_prompt(key: str) -> str
    def get_instructions(instruction_type: str) -> List[str]
    def get_template(key: str) -> str
    def format_template(template_key: str, **kwargs) -> str
```

#### PromptBuilder (models/prompts.py)
```python
class PromptBuilder:
    def __init__(self)
    def build_chat_prompt(self, query_text: str, contexts: List[str], images: List[Dict], chat_history: List[Dict], is_technical: bool = False) -> str
    def build_messages(self, prompt: str) -> List[Dict[str, str]]
    def build_no_answer_message(self, query_text: str) -> List[Dict[str, str]]
```

### Utility Components

#### ImageStore (utils/image_store.py)
```python
class ImageStore:
    def __init__(self, base_path: Path)
    def store_image(self, image: Image.Image, source_doc: str, page_num: int, caption: Optional[str] = None, context: Optional[str] = None) -> str
    def get_image(self, image_id: str) -> Tuple[Optional[Image.Image], Optional[Dict]]
    def get_base64_image(self, image_id: str) -> Optional[str]
    def delete_image(self, image_id: str) -> bool
```

#### FAISS Utils (utils/FAISS_utils.py)
```python
def initialize_faiss_index(dimension: int, use_gpu: bool = False) -> faiss.Index
def add_to_faiss(embedding, source_file_name, content_type, content, index, metadata)
def query_faiss(index, metadata, query_embeddings, top_k)
def query_with_context(index, metadata, model, processor, device, text_query=None, image_query=None, top_k=5)
```

#### Document Utils (utils/document_utils.py)
```python
def rescan_documents(config: CONFIG) -> tuple[bool, str]
def remove_document_from_rag(doc_path: Path) -> tuple[bool, str]
def update_processed_files_list(file_path: Path, remove: bool = False)
def validate_folder_name(name: str) -> tuple[bool, str]
def create_folder(parent_path: Path, folder_name: str) -> tuple[bool, str]
```

#### RAG Utils (utils/RAG_utils.py)
```python
def extract_text_and_images_from_pdf(pdf_path) -> Tuple[str, List[Dict]]
def extract_text_and_images_from_word(doc_path) -> Tuple[str, List[Dict]]
def extract_text_and_images_from_excel(excel_path) -> Tuple[str, List[Dict]]
def chunk_text(text: str, source_path: str, chunk_size=CONFIG.CHUNK_SIZE) -> List[Dict]
```

#### LLM Utils (utils/LLM_utils.py)
```python
def CLIP_init(model_name: str = "openai/clip-vit-base-patch32") -> Tuple[Model, Processor, str]
def encode_with_clip(texts: List[str], images: List[Image.Image], model, processor, device) -> Tuple[np.ndarray, np.ndarray]
def openai_post_request(messages: List[Dict], model_name: str, max_tokens: int, temperature: float, api_key: str) -> Dict
```

## Data Models

### Query Models
```python
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryType(BaseModel):
    is_overview: bool = False
    is_technical: bool = False
    is_summary: bool = False

class QueryResponse(BaseModel):
    text_response: str
    images: List[Dict[str, Any]] = field(default_factory=list)
```

### Configuration Model
```python
@dataclass
class Config:
    SERVER_PORT: int = 9000
    RAW_DOCUMENTS_PATH: Path = Path("Raw Documents")
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    USE_GPU: bool = True
    # ... other configuration parameters
```

## API Endpoints

### Document Management
```python
@app.post("/upload/document")
@app.post("/process/documents")
@app.get("/get/documents")
@app.delete("/delete/document")
@app.post("/folder/create")
@app.delete("/folder/delete")
@app.put("/folder/rename")
```

### Query Processing
```python
@app.post("/query/text")
@app.post("/query/image")
@app.post("/chat/reset")
@app.get("/chat/history")
```

### System Management
```python
@app.post("/webhook")
@app.post("/rescan")
```

## Event Flow

1. Document Upload
```mermaid
graph LR
    A[Upload] --> B[Store File]
    B --> C[Process Document]
    C --> D[Extract Text/Images]
    D --> E[Generate Embeddings]
    E --> F[Update Index]
```

2. Query Processing
```mermaid
graph LR
    A[Query] --> B[Get Context]
    B --> C[Build Prompt]
    C --> D[Get LLM Response]
    D --> E[Format Response]
```

