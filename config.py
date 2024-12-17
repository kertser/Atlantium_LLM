
from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class Config:
    # Ports and URLs
    SERVER_PORT: int = 9000

    # Paths
    RAW_DOCUMENTS_PATH: Path = Path("Raw Documents")

    INDICES: str = "RAG_Data/indices"
    RAG_DATA:  str = "RAG_Data"
    FAISS_INDEX_PATH: Path = Path("RAG_Data/indices/faiss_index.bin")
    METADATA_PATH: Path = Path("RAG_Data/indices/faiss_metadata.json")
    STORED_TEXT_CHUNKS_PATH: Path = Path("RAG_Data/stored_text_chunks")

    LOG_PATH: Path = Path("system.log")
    STORED_IMAGES_PATH: Path = Path("RAG_Data/stored_images")

    # CLIP Configuration
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSION: int = 512
    USE_GPU: bool = True

    # Document Processing
    BATCH_SIZE: int = 5  # Document processing in batches. For limited RAM it is 2-5. For GPU 8-16GB it is 8-16
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_SIZE: int = 100
    CHUNK_SIZE: int = 1000  # Optimal for larger content. Smaller chunks are more selective, but harder to compare
    SIMILARITY_THRESHOLD: float = 0.75  # Text similarity
    IMAGE_SIMILARITY_THRESHOLD: float = 0.25  # Image similarity
    TECHNICAL_CONFIDENCE_THRESHOLD: float = 0.6  # Technical confidence
    MAX_METADATA_SIZE: int = 10000000  # We have to keep it large, since 100-200 docs can take 15Gb space
    SUPPORTED_EXTENSIONS: List[str] = None

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

        # Ensure paths are Path objects
        self.RAW_DOCUMENTS_PATH = Path(self.RAW_DOCUMENTS_PATH)
        self.FAISS_INDEX_PATH = Path(self.FAISS_INDEX_PATH)
        self.METADATA_PATH = Path(self.METADATA_PATH)
        self.LOG_PATH = Path(self.LOG_PATH)


# Create global config instance
CONFIG = Config()
