"""Module for initializing/resetting RAG database and related files."""

import shutil
from pathlib import Path
import logging
from typing import List, Set, Dict
import os
from config import CONFIG

def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def setup_logging() -> None:
    """Set up console-only logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

def ensure_root_directory() -> None:
    """Ensure we're in the project root directory."""
    project_root = get_project_root()
    current_dir = Path.cwd()

    if current_dir != project_root:
        os.chdir(project_root)
        logging.info(f"Changed working directory to project root: {project_root}")

def create_required_directories() -> Dict[str, bool]:
    """
    Create all required directories from project root.

    Returns:
        Dict[str, bool]: Dictionary of directory paths and their creation status
    """
    ensure_root_directory()
    creation_status = {}

    directories = {
        "Raw Documents": CONFIG.RAW_DOCUMENTS_PATH,
        "RAG Data": CONFIG.RAG_DATA,
        "Stored Images": CONFIG.STORED_IMAGES_PATH,
        "Text Chunks": CONFIG.STORED_TEXT_CHUNKS_PATH,
        "Logs": CONFIG.LOG_PATH
    }

    logging.info("Starting directory creation process...")

    for name, directory in directories.items():
        try:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                creation_status[str(directory)] = True
                logging.info(f"Created directory: {name} at {directory}")
            else:
                creation_status[str(directory)] = False
                logging.info(f"Directory already exists: {name} at {directory}")

            # Verify directory is writable
            test_file = directory / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
                logging.info(f"Verified write access to: {directory}")
            except Exception as e:
                logging.error(f"Directory {directory} is not writable: {e}")
                creation_status[str(directory)] = False

        except Exception as e:
            logging.error(f"Failed to create/verify directory {name} at {directory}: {e}")
            creation_status[str(directory)] = False

    # Log summary
    created = sum(1 for status in creation_status.values() if status)
    existing = len(creation_status) - created
    logging.info(f"Directory creation summary: {created} created, {existing} already existed")

    return creation_status

def initialize_rag_database(
        paths_to_clean: List[Path] = None,
        directories_to_clean: List[Path] = None,
        clean_raw_documents: bool = False
) -> None:
    """Initialize/reset the RAG database and create required directory structure."""
    try:
        # Set up logging first
        setup_logging()

        # Ensure we're in project root
        ensure_root_directory()
        project_root = get_project_root()
        logging.info(f"Initializing RAG database in: {project_root}")

        # Define default paths relative to project root
        if paths_to_clean is None:
            paths_to_clean = [
                CONFIG.FAISS_INDEX_PATH,
                CONFIG.METADATA_PATH,
                CONFIG.IMAGE_METADATA_PATH,
                Path("processed_files.json"),
            ]

        if directories_to_clean is None:
            directories_to_clean = [
                CONFIG.RAG_DATA,
                CONFIG.STORED_TEXT_CHUNKS_PATH,
                CONFIG.STORED_IMAGES_PATH,
                CONFIG.LOG_PATH
            ]

        # Track which paths we've handled
        handled_paths: Set[Path] = set()

        # Remove specific files
        logging.info("Cleaning existing files...")
        for file_path in paths_to_clean:
            if file_path in handled_paths:
                continue

            abs_path = file_path if file_path.is_absolute() else project_root / file_path
            try:
                if abs_path.exists():
                    abs_path.unlink()
                    logging.info(f"Removed file: {abs_path}")
                else:
                    logging.info(f"File does not exist (skipping): {abs_path}")
                handled_paths.add(file_path)
            except Exception as e:
                logging.error(f"Error removing file {abs_path}: {e}")

        # Clean directories
        logging.info("Cleaning existing directories...")
        for dir_path in directories_to_clean:
            if dir_path in handled_paths:
                continue

            abs_path = dir_path if dir_path.is_absolute() else project_root / dir_path
            try:
                if abs_path.exists():
                    shutil.rmtree(abs_path)
                    logging.info(f"Removed directory and contents: {abs_path}")
                else:
                    logging.info(f"Directory does not exist (skipping): {abs_path}")
            except Exception as e:
                logging.error(f"Error removing directory {abs_path}: {e}")

        # Create required directories
        logging.info("Creating required directory structure...")
        dir_status = create_required_directories()

        # Create empty processed_files.json
        try:
            processed_files_path = project_root / "processed_files.json"
            processed_files_path.write_text("[]", encoding="utf-8")
            logging.info("Created empty processed_files.json")
        except Exception as e:
            logging.error(f"Error creating processed_files.json: {e}")

        # Log final status
        logging.info("\nInitialization Summary:")
        logging.info("------------------------")
        logging.info("Directory Status:")
        for path, was_created in dir_status.items():
            status = "Created" if was_created else "Already existed"
            logging.info(f"- {path}: {status}")
        logging.info("------------------------")
        logging.info("RAG database initialization completed successfully")

    except Exception as e:
        logging.error(f"Critical error during initialization: {e}")
        raise

if __name__ == "__main__":
    initialize_rag_database()