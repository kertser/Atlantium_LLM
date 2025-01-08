import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Set

# Get base directory from environment variable or use current directory for local development
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # BASE_DIR as a class attribute
    BASE_DIR: Path = BASE_DIR
    # Ports and URLs
    SERVER_PORT: int = 9000

    # Base Paths
    RAW_DOCUMENTS_PATH: Path = BASE_DIR / "Raw Documents"
    RAG_DATA: Path = BASE_DIR / "RAG_Data"  # Base directory for all RAG data

    # Specific Data Paths - use RAG_DATA as base
    FAISS_INDEX_PATH: Path = RAG_DATA / "faiss_index.bin"
    METADATA_PATH: Path = RAG_DATA / "faiss_metadata.json"
    IMAGE_METADATA_PATH: Path = RAG_DATA / "image_metadata.json"
    STORED_IMAGES_PATH: Path = RAG_DATA / "stored_images"
    STORED_TEXT_CHUNKS_PATH: Path = RAG_DATA / "stored_text_chunks"
    PROCESSED_FILES_PATH: Path = BASE_DIR / "processed_files.json"

    # Logging
    LOG_PATH: Path = BASE_DIR / "logs"
    LOG_BACKUP_COUNT: int = 5  # Maximum log backups
    MAX_LOG_SIZE: int = 10000

    # Aggegator Configuration
    AGGREGATOR_MODEL: str = "gpt-4o-mini"
    AGGREGATOR_TEMPERATURE: float = 0.1
    AGGREGATOR_MAX_TOKENS: int = 2000

    # WEBSEARCH Configuration
    WEBSEARCH_MODEL: str = "gpt-4o-mini"
    WEBSEARCH_MAX_RESULTS: int = 10

    # RED Calculator Configuration
    RED_CALCULATOR_MODEL: str = "gpt-4o-mini"
    RED_CALCULATOR_TEMPERATURE: float = 0.1
    RED_CALCULATOR_MAX_TOKENS: int = 2000
    RED_CALCULATOR_DEFAULT_DRIVE: float = 100
    RED_CALCULATOR_DEFAULT_EFFICIENCY: float = 80 # May be set to 100% by default.

    # CLIP Configuration
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSION: int = 512
    USE_GPU: bool = True

    # Thresholds (not percentiles)
    SIMILARITY_THRESHOLD: float = 0.65  # Text similarity (it shall be 0.6-0.8)
    IMAGE_SIMILARITY_THRESHOLD: float = 0.35  # Image similarity - set to high value to avoid false positives
    TECHNICAL_CONFIDENCE_THRESHOLD: float = 0.75  # Technical confidence
    DEDUPLICATION_THRESHOLD = 0.70  # Threshold for image deduplication

    # Query Configuration
    DEFAULT_TOP_K: int = 20
    TEMPERATURE: float = 0
    GPT_MODEL: str = "gpt-4o-mini"

    # Vision model settings
    GPT_VISION_MODEL: str = "gpt-4o"
    VISION_MAX_TOKENS: int = 4096
    VISION_QUALITY: str = "auto"

    # Response formatting
    DEFAULT_LINE_LENGTH: int = 80
    BULLET_INDENT: int = 2

    # Chat parameters
    MAX_CHAT_HISTORY: int = 3

    # Token limits for completeness
    MAX_TOKENS: int = 3000  # General setting
    SUMMARY_MAX_TOKENS: int = 2000  # Setting for summaries
    DETAIL_MAX_TOKENS: int = 4000  # Setting for detailed responses

    # Document Processing
    BATCH_SIZE: int = 5  # Document processing in batches. For limited RAM it is 2-5. For GPU 8-16GB it is 8-16
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 50
    CHUNK_SIZE: int = 300  # Smaller chunks are more selective, but harder to compare
    SUPPORTED_EXTENSIONS: List[str] = None
    MAX_TEXT_LENGTH: int = 10000  # Maximum length of stored text chunks
    MAX_METADATA_SIZE: int = 1000000  # Maximum size in bytes
    METADATA_TEXT_LIMIT:  int = 1500  # Maximum text length in metadata entries
    COMPRESSION_ENABLED = True
    CLEANUP_FREQUENCY = 10  # Cleanup every N batches

    # Image Processing
    MIN_IMAGE_SIZE: int = 150  # Leave as is for basic filtering
    MIN_ICON_SIZE: int = 100  # Leave as is for icon filtering
    MAX_CONTEXT_RANGE: int = 100  # Leave as is for text context
    MAX_ASPECT_RATIO: int = 5  # Maximum width/height ratio

    # Image Quality Settings
    IMAGE_DPI: Tuple[int, int] = (300, 300)  # DPI
    IMAGE_BITS: int = 32  # Good for preserving color depth
    COMPRESSION_LEVEL: int = 0  # No compression - good for quality
    IMAGE_QUALITY: int = 100  # Maximum quality - perfect

    # Image Enhancement Settings
    SHARPEN_FACTOR: float = 1.1  # Reduce sharpening to prevent artifacts
    COLOR_FACTOR: float = 1.0  # Keep at 1.0 to preserve original colors
    CONTRAST_FACTOR: float = 1.0  # Set to 1.0 to preserve original contrast

    # Save Format Settings
    PREFERRED_SAVE_FORMAT: str = 'PNG'  # Good choice for lossless quality
    VALID_IMAGE_MODES: List[str] = field(
        default_factory=lambda: ['RGB', 'RGBA', 'L', 'LA', 'P', '1', 'I']
    )
    SUPPORTED_IMAGE_FORMATS: Set[str] = field(
        default_factory=lambda: {'PNG', 'JPEG', 'JPG', 'BMP', 'TIFF', 'GIF'}
    )
    SUPPORTED_EXTENSIONS: List[str] = field(
        default_factory=lambda: ['.pdf', '.docx', '.xlsx']
    )

    def validate_metadata_size(self, metadata_path):
        if os.path.getsize(metadata_path) > self.MAX_METADATA_SIZE:
            # Trigger cleanup
            return False
        return True

    def __post_init__(self):
        if self.SUPPORTED_EXTENSIONS is None:
            self.SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx']

        # Create directories if they don't exist
        for path in [self.RAW_DOCUMENTS_PATH, self.RAG_DATA, self.LOG_PATH,
                     self.STORED_IMAGES_PATH, self.STORED_TEXT_CHUNKS_PATH]:
            path.mkdir(parents=True, exist_ok=True)

            # Fix mutable defaults using field(default_factory=...)
            VALID_IMAGE_MODES: List[str] = field(
                default_factory=lambda: ['RGB', 'RGBA', 'L', 'LA', 'P', '1', 'I']
            )

            SUPPORTED_IMAGE_FORMATS: Set[str] = field(
                default_factory=lambda: {'PNG', 'JPEG', 'JPG', 'BMP', 'TIFF', 'GIF'}
            )

            SUPPORTED_EXTENSIONS: List[str] = field(
                default_factory=lambda: ['.pdf', '.docx', '.xlsx']
            )

            def __post_init__(self):
                # Create directories if they don't exist
                for path in [self.RAW_DOCUMENTS_PATH, self.RAG_DATA, self.LOG_PATH,
                             self.STORED_IMAGES_PATH, self.STORED_TEXT_CHUNKS_PATH]:
                    path.mkdir(parents=True, exist_ok=True)

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
