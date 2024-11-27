from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass
class Config:
    # Paths
    RAW_DOCUMENTS_PATH: Path = Path("Raw Documents")
    FAISS_INDEX_PATH: Path = Path("faiss_index.bin")
    METADATA_PATH: Path = Path("faiss_metadata.json")
    LOG_PATH: Path = Path("system.log")

    # CLIP Configuration
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSION: int = 512
    USE_GPU: bool = True

    # Document Processing
    CHUNK_SIZE: int = 512
    SUPPORTED_EXTENSIONS: List[str] = None

    # Query Configuration
    DEFAULT_TOP_K: int = 5
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.1
    GPT_MODEL: str = "gpt-4-turbo"
    GPT_VISION_MODEL: str = "gpt-4o"

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