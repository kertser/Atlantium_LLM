
from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class Config:
    # Ports and URLs
    SERVER_PORT: int = 9000

    # Base Paths
    RAW_DOCUMENTS_PATH: Path = Path("Raw Documents")
    RAG_DATA: Path = Path("RAG_Data")  # Base directory for all RAG data

    # Specific Data Paths
    FAISS_INDEX_PATH: Path = Path("RAG_Data/faiss_index.bin")
    METADATA_PATH: Path = Path("RAG_Data/faiss_metadata.json")
    IMAGE_METADATA_PATH: Path = Path("RAG_Data/image_metadata.json")
    STORED_IMAGES_PATH: Path = Path("RAG_Data/stored_images")
    STORED_TEXT_CHUNKS_PATH: Path = Path("RAG_Data/stored_text_chunks")

    # Logging
    LOG_PATH: Path = Path("logs")
    LOG_BACKUP_COUNT:  int = 5  # Maximum log backups
    MAX_LOG_SIZE: int = 10000

    # CLIP Configuration
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSION: int = 512
    USE_GPU: bool = True

    # Document Processing
    BATCH_SIZE: int = 5  # Document processing in batches. For limited RAM it is 2-5. For GPU 8-16GB it is 8-16
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_SIZE: int = 100
    CHUNK_SIZE: int = 1000  # Optimal for larger content. Smaller chunks are more selective, but harder to compare
    SIMILARITY_THRESHOLD: float = 0.8  # Text similarity
    IMAGE_SIMILARITY_THRESHOLD: float = 0.25  # Image similarity
    TECHNICAL_CONFIDENCE_THRESHOLD: float = 0.6  # Technical confidence
    SUPPORTED_EXTENSIONS: List[str] = None
    MAX_TEXT_LENGTH: int = 10000  # Maximum length of stored text chunks
    MAX_METADATA_SIZE: int = 1000000  # Maximum size in bytes
    COMPRESSION_ENABLED = True
    CLEANUP_FREQUENCY = 10  # Cleanup every N batches

    def validate_metadata_size(self, metadata_path):
        if os.path.getsize(metadata_path) > self.MAX_METADATA_SIZE:
            # Trigger cleanup
            return False
        return True

    # Token limits for completeness
    MAX_TOKENS: int = 1000
    SUMMARY_MAX_TOKENS: int = 500  # Setting for summaries
    DETAIL_MAX_TOKENS: int = 1500  # Setting for detailed responses

    # Query Configuration
    DEFAULT_TOP_K: int = 5
    TEMPERATURE: float = 0.01
    GPT_MODEL: str = "gpt-4o-mini"

    # Vision model settings
    GPT_VISION_MODEL: str = "gpt-4o"
    VISION_MAX_TOKENS: int = 500
    VISION_QUALITY: str = "auto"

    # Response formatting
    DEFAULT_LINE_LENGTH: int = 80
    BULLET_INDENT: int = 2

    # Chat parameters
    MAX_CHAT_HISTORY: int = 3

    def __post_init__(self):
        if self.SUPPORTED_EXTENSIONS is None:
            self.SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx']

            # Ensure all paths are Path objects
            self.RAW_DOCUMENTS_PATH = Path(self.RAW_DOCUMENTS_PATH)
            self.RAG_DATA = Path(self.RAG_DATA)
            self.FAISS_INDEX_PATH = Path(self.FAISS_INDEX_PATH)
            self.METADATA_PATH = Path(self.METADATA_PATH)
            self.IMAGE_METADATA_PATH = Path(self.IMAGE_METADATA_PATH)
            self.STORED_IMAGES_PATH = Path(self.STORED_IMAGES_PATH)
            self.STORED_TEXT_CHUNKS_PATH = Path(self.STORED_TEXT_CHUNKS_PATH)
            self.LOG_PATH = Path(self.LOG_PATH)

# Create global config instance
CONFIG = Config()
