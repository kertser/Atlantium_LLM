"""Module for initializing/resetting RAG database and related files."""

import shutil
from pathlib import Path
import logging
from typing import List, Set
import os

from config import CONFIG


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def initialize_rag_database(
    paths_to_clean: List[Path] = None,
    directories_to_clean: List[Path] = None,
    clean_raw_documents: bool = False
) -> None:
    """
    Initialize/reset the RAG database by cleaning all indices and generated files.

    Args:
        paths_to_clean: List of specific files to remove
        directories_to_clean: List of directories to clean/recreate
        clean_raw_documents: If True, also cleans the Raw Documents directory. Default is False.
    """
    # Change to project root directory
    project_root = get_project_root()
    os.chdir(project_root)

    # Define default paths relative to project root
    if paths_to_clean is None:
        paths_to_clean = [
            CONFIG.FAISS_INDEX_PATH,
            CONFIG.METADATA_PATH,
            CONFIG.STORED_IMAGES_PATH,
            Path("processed_files.json"),
            Path("system.log")
        ]

    if directories_to_clean is None:
        directories_to_clean = [
            CONFIG.STORED_IMAGES_PATH,
            Path("indices"),
            Path("RAG_Data")
        ]

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Track which paths we've handled to avoid duplicates
        handled_paths: Set[Path] = set()

        logger.info(f"Starting RAG database initialization in {project_root}...")

        # Remove specific files
        for file_path in paths_to_clean:
            if file_path in handled_paths:
                continue

            abs_path = project_root / file_path
            try:
                if abs_path.exists():
                    abs_path.unlink()
                    logger.info(f"Removed file: {abs_path}")
                handled_paths.add(file_path)
            except Exception as e:
                logger.error(f"Error removing file {abs_path}: {e}")

        # Add Raw Documents to directories_to_clean if specified
        dirs_to_process = list(directories_to_clean)
        if clean_raw_documents:
            dirs_to_process.append(CONFIG.RAW_DOCUMENTS_PATH)
            logger.info("Raw Documents directory will be cleaned")

        # Clean and recreate directories
        for dir_path in dirs_to_process:
            if dir_path in handled_paths:
                continue

            abs_path = project_root / dir_path
            try:
                if abs_path.exists():
                    shutil.rmtree(abs_path)
                    logger.info(f"Removed directory: {abs_path}")
                abs_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {abs_path}")
                handled_paths.add(dir_path)
            except Exception as e:
                logger.error(f"Error handling directory {abs_path}: {e}")

        # Always ensure Raw Documents directory exists
        raw_docs_path = project_root / CONFIG.RAW_DOCUMENTS_PATH
        raw_docs_path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured Raw Documents directory exists")

        # Create empty processed_files.json
        try:
            processed_files_path = project_root / "processed_files.json"
            processed_files_path.write_text("[]", encoding="utf-8")
            logger.info("Created empty processed_files.json")
        except Exception as e:
            logger.error(f"Error creating processed_files.json: {e}")

        # Create required subdirectories
        try:
            images_path = project_root / CONFIG.STORED_IMAGES_PATH / "images"
            images_path.mkdir(parents=True, exist_ok=True)
            logger.info("Created images subdirectory")
        except Exception as e:
            logger.error(f"Error creating images subdirectory: {e}")

        logger.info("RAG database initialization completed successfully")

    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        raise


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    # Example: python -m utils.initialize_RAG
    initialize_rag_database()
