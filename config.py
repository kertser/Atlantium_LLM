from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass
class Config:
    # Ports and URLs
    SERVER_PORT: int = 9000

    # Paths
    RAW_DOCUMENTS_PATH: Path = Path("Raw Documents")
    FAISS_INDEX_PATH: Path = Path("faiss_index.bin")
    METADATA_PATH: Path = Path("faiss_metadata.json")
    LOG_PATH: Path = Path("system.log")
    STORED_IMAGES_PATH: Path = Path("RAG_Data/stored_images")

    # CLIP Configuration
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSION: int = 512
    USE_GPU: bool = True

    # Document Processing
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100
    SIMILARITY_THRESHOLD: float = 0.4
    IMAGE_SIMILARITY_THRESHOLD: float = 0.3
    MAX_METADATA_SIZE: int = 10000
    CHUNK_SIZE: int = 512
    SUPPORTED_EXTENSIONS: List[str] = None

    # Token limits for completeness
    MAX_TOKENS: int = 1000
    SUMMARY_MAX_TOKENS: int = 500  # Setting for summaries
    DETAIL_MAX_TOKENS: int = 1500  # Setting for detailed responses

    # Query Configuration
    DEFAULT_TOP_K: int = 3
    TEMPERATURE: float = 0.1
    GPT_MODEL: str = "gpt-4o-mini"

    # Vision model settings
    GPT_VISION_MODEL: str = "gpt-4o"
    VISION_MAX_TOKENS: int = 500
    VISION_QUALITY: str = "auto"

    # Response formatting
    DEFAULT_LINE_LENGTH: int = 80
    BULLET_INDENT: int = 2

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