import os
import platform
import subprocess
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Set, Tuple
import shutil
import faiss
import json
import numpy as np
from config import CONFIG
from utils.FAISS_utils import load_faiss_index, load_metadata, save_faiss_index, save_metadata
from utils.image_store import ImageStore

logger = logging.getLogger(__name__)


def sanitize_filename(filepath: Path) -> Tuple[Path, bool]:
    """
    Sanitize filename by removing leading/trailing spaces in filename and before extension.

    Args:
        filepath: Path object of the file

    Returns:
        Tuple of (new_path, was_renamed)
    """
    original_path = filepath
    parent = filepath.parent
    filename = filepath.name

    # Split filename and extension
    name_parts = filename.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        # Remove leading/trailing spaces from name
        new_name = f"{name.strip()}.{ext.strip()}"
    else:
        # No extension
        new_name = filename.strip()

    new_path = parent / new_name

    if new_path != original_path:
        try:
            original_path.rename(new_path)
            logging.info(f"Renamed file from '{original_path}' to '{new_path}'")
            return new_path, True
        except Exception as e:
            logging.error(f"Error renaming file '{original_path}': {e}")
            return original_path, False

    return original_path, False


def sanitize_uploaded_file(file_path: Path) -> Path:
    """
    Sanitize an uploaded file's name before saving.

    Args:
        file_path: Path object of the uploaded file

    Returns:
        Path object with sanitized name
    """
    parent = file_path.parent
    filename = file_path.name

    # Split filename and extension
    name_parts = filename.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        sanitized_name = f"{name.strip()}.{ext.strip()}"
    else:
        sanitized_name = filename.strip()

    return parent / sanitized_name


def compare_and_update_rag(raw_docs_path: Path, processed_files: Set[str], supported_extensions: Set[str]) -> Tuple[
    List[Path], List[Path]]:
    """
    Compare raw documents with processed files and identify new and removed documents.

    Args:
        raw_docs_path: Path to raw documents directory
        processed_files: Set of processed file paths
        supported_extensions: Set of supported file extensions

    Returns:
        Tuple of (new_files, removed_files)
    """
    # Get current documents
    current_docs = set()
    for ext in supported_extensions:
        for doc_path in raw_docs_path.rglob(f"*{ext}"):
            # Sanitize filename if needed
            doc_path, was_renamed = sanitize_filename(doc_path)
            current_docs.add(str(doc_path.absolute()))

    # Find new and removed files
    new_files = [Path(p) for p in current_docs - processed_files]
    removed_files = [Path(p) for p in processed_files - current_docs]

    return new_files, removed_files

def get_document_count(base_path: Path, supported_extensions: Set[str]) -> int:
    """
    Count only document files (excluding folders) in the given path.

    Args:
        base_path: Base directory to search
        supported_extensions: Set of supported file extensions

    Returns:
        Number of documents found
    """
    count = 0
    for item in base_path.rglob("*"):
        if item.is_file() and item.suffix.lower() in supported_extensions:
            count += 1
    return count


def rescan_documents(config: CONFIG) -> tuple[bool, str]:
    """
    Rescan documents and update RAG system by comparing raw documents with processed ones.

    Args:
        config: Application configuration object

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        logger.info("Starting document rescan process")

        # Load processed files list
        processed_files_path = Path("processed_files.json")
        if processed_files_path.exists():
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()

        # Convert supported extensions to set of lowercase extensions with dots
        supported_extensions = {f".{ext.lower().lstrip('.')}" for ext in config.SUPPORTED_EXTENSIONS}

        # Compare current files with processed files
        new_files, removed_files = compare_and_update_rag(
            raw_docs_path=config.RAW_DOCUMENTS_PATH,
            processed_files=processed_files,
            supported_extensions=supported_extensions
        )

        if not new_files and not removed_files:
            logger.info("No changes detected in document collection")
            return True, "No changes detected"

        # Process new files if any
        if new_files:
            logger.info(f"Found {len(new_files)} new documents to process")
            try:
                # Initialize required components
                from utils.LLM_utils import CLIP_init
                from utils.FAISS_utils import initialize_faiss_index, load_faiss_index, load_metadata
                from utils.image_store import ImageStore
                from RAG_processor import process_documents

                # Load or initialize CLIP model
                model, processor, device = CLIP_init(config.CLIP_MODEL_NAME)

                # Load or initialize FAISS index and metadata
                try:
                    index = load_faiss_index(config.FAISS_INDEX_PATH)
                    metadata = load_metadata(config.METADATA_PATH)
                except:
                    index = initialize_faiss_index(config.EMBEDDING_DIMENSION, config.USE_GPU)
                    metadata = []

                # Initialize image store
                image_store = ImageStore(config.STORED_IMAGES_PATH)

                # Process documents with all required arguments
                success = process_documents(
                    model=model,
                    processor=processor,
                    device=device,
                    index=index,
                    metadata=metadata,
                    image_store=image_store,
                    doc_paths=[str(p) for p in new_files]
                )

                if not success:
                    raise Exception("Failed to process new documents")

            except Exception as e:
                logger.error(f"Error processing new documents: {e}")
                return False, f"Failed to process new documents: {str(e)}"

        # Remove documents that no longer exist
        if removed_files:
            logger.info(f"Found {len(removed_files)} documents to remove from RAG")
            failed_removals = []
            for file_path in removed_files:
                success, msg = remove_document_from_rag(file_path)
                if not success:
                    failed_removals.append(f"{file_path}: {msg}")
                    logger.error(f"Failed to remove document {file_path}: {msg}")

            if failed_removals:
                return False, f"Failed to remove some documents: {'; '.join(failed_removals)}"

        total_changes = len(new_files) + len(removed_files)
        success_msg = f"Rescan completed: {len(new_files)} new documents processed, {len(removed_files)} documents removed"
        logger.info(success_msg)
        return True, success_msg

    except Exception as e:
        error_msg = f"Error during rescan: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def open_file_with_default_program(file_path: str) -> tuple[bool, str]:
    """
    Opens a file with the default system program.

    Args:
        file_path: Path to the file to open

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        file_path = str(file_path)  # Ensure string path
        system = platform.system()

        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # Linux/Unix
            subprocess.run(["xdg-open", file_path], check=True)

        return True, "File opened successfully"

    except FileNotFoundError:
        return False, "File not found"
    except PermissionError:
        return False, "Permission denied"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to open file: {str(e)}"
    except Exception as e:
        logger.error(f"Error opening file {file_path}: {str(e)}")
        return False, f"Unexpected error: {str(e)}"


def remove_document_from_rag(doc_path: Path) -> tuple[bool, str]:
    """
    Removes a document and its associated data from the RAG system.
    Uses direct vector copying instead of reconstruction.
    """
    try:
        logger.info(f"Starting removal of document: {doc_path}")

        # Load current data
        original_index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
        metadata = load_metadata(CONFIG.METADATA_PATH)
        image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)
        logger.info(f"Successfully loaded index with {len(metadata)} entries")

        # Get relative path for comparison
        try:
            relative_path = doc_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH)
        except ValueError:
            relative_path = doc_path
        logger.info(f"Using relative path: {relative_path}")

        # Identify entries to remove and filter metadata
        indices_to_remove = set()
        new_metadata = []

        for idx, entry in enumerate(metadata):
            entry_path = Path(entry.get('path', ''))
            if entry_path == relative_path or entry_path.name == doc_path.name:
                indices_to_remove.add(idx)
                logger.info(f"Found matching entry at index {idx}: {entry_path}")

                # Handle file removal
                if entry.get('type') == 'image':
                    image_id = entry.get('image', {}).get('id')
                    if image_id:
                        image_store.delete_image(image_id)
                        logger.info(f"Deleted image {image_id}")
                elif entry.get('type') == 'text-chunk':
                    chunk_path = entry.get('chunk')
                    if chunk_path:
                        try:
                            chunk_file = CONFIG.STORED_TEXT_CHUNKS_PATH / chunk_path
                            if chunk_file.exists():
                                chunk_file.unlink()
                                logger.info(f"Deleted chunk file: {chunk_path}")
                        except Exception as e:
                            logger.error(f"Error deleting chunk file: {str(e)}")
            else:
                new_metadata.append(entry)

        if not indices_to_remove:
            logger.info(f"Document {doc_path.name} not found in RAG system")
            return True, "Document not found in RAG system"

        # Remove chunk directory
        chunk_dir = CONFIG.STORED_TEXT_CHUNKS_PATH / Path(relative_path).stem
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
            logger.info(f"Removed chunk directory: {chunk_dir}")

        # Save updated metadata
        save_metadata(new_metadata, CONFIG.METADATA_PATH)
        logger.info(f"Saved updated metadata with {len(new_metadata)} entries")

        # Create new index and copy vectors
        new_index = faiss.IndexFlatL2(original_index.d)
        batch_size = 1000  # Process vectors in batches

        for start_idx in range(0, original_index.ntotal, batch_size):
            end_idx = min(start_idx + batch_size, original_index.ntotal)
            batch_indices = range(start_idx, end_idx)

            # Filter out indices to remove from this batch
            valid_indices = [idx for idx in batch_indices if idx not in indices_to_remove]

            if valid_indices:
                try:
                    # Get vectors directly from the original index
                    vectors = original_index.reconstruct_batch(valid_indices)
                    new_index.add(vectors)
                except Exception as e:
                    logger.error(f"Error processing batch {start_idx}-{end_idx}: {str(e)}")

        # Save updated index
        save_faiss_index(new_index, CONFIG.FAISS_INDEX_PATH)
        logger.info(f"Saved updated index with {new_index.ntotal} vectors")

        # Update processed files list
        update_processed_files_list(doc_path, remove=True)
        logger.info("Updated processed files list")

        return True, f"Successfully removed document with {len(indices_to_remove)} entries"

    except Exception as e:
        error_msg = f"Failed to remove document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def update_processed_files_list(file_path: Path, remove: bool = False) -> None:
    """Updates the processed_files.json list."""
    try:
        processed_files_path = Path("processed_files.json")
        if processed_files_path.exists():
            with open(processed_files_path, 'r') as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()

        str_path = str(file_path.absolute())
        if remove:
            processed_files.discard(str_path)
        else:
            processed_files.add(str_path)

        with open(processed_files_path, 'w') as f:
            json.dump(list(processed_files), f, indent=2)

    except Exception as e:
        logger.error(f"Error updating processed files list: {e}")
        raise


def delete_folder_from_rag(folder_path: Path) -> tuple[bool, str, List[str]]:
    """
    Recursively deletes a folder and removes all its documents from RAG.

    Args:
        folder_path: Path to the folder to delete

    Returns:
        Tuple of (success: bool, message: str, list of errors)
    """
    errors = []
    try:
        # Get all documents in folder and subfolders
        documents = []
        for ext in CONFIG.SUPPORTED_EXTENSIONS:
            documents.extend(folder_path.rglob(f"*{ext}"))

        # Remove each document from RAG
        for doc in documents:
            success, msg = remove_document_from_rag(doc)
            if not success:
                errors.append(f"Failed to remove {doc}: {msg}")

        # Delete the folder and its contents
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            errors.append(f"Failed to delete folder: {str(e)}")
            return False, "Failed to delete folder", errors

        return len(errors) == 0, "Folder deleted successfully" if len(
            errors) == 0 else "Folder deleted with errors", errors

    except Exception as e:
        logger.error(f"Error deleting folder {folder_path}: {str(e)}")
        return False, f"Failed to delete folder: {str(e)}", errors


def rename_folder_in_rag(old_path: Path, new_path: Path) -> tuple[bool, str]:
    """
    Renames a folder and updates all RAG references using recursive traversal.

    Args:
        old_path: Current folder path
        new_path: New folder path

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Load all metadata files
        faiss_metadata_path = CONFIG.METADATA_PATH
        image_metadata_path = CONFIG.STORED_IMAGES_PATH / "image_metadata.json"
        processed_files_path = Path("processed_files.json")

        # Get the old and new folder names for replacement
        old_folder_name = old_path.name
        new_folder_name = new_path.name

        def replace_folder_name(value: str) -> str:
            """Replace old folder name with new folder name in a string."""
            if not isinstance(value, str):
                return value
            return value.replace(old_folder_name, new_folder_name)

        def update_recursively(obj: Any) -> Any:
            """
            Recursively traverse through data structure and replace folder names.

            Args:
                obj: The object to traverse (dict, list, or primitive type)

            Returns:
                Updated object with replaced folder names
            """
            if isinstance(obj, dict):
                return {key: update_recursively(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [update_recursively(item) for item in obj]
            elif isinstance(obj, str):
                return replace_folder_name(obj)
            return obj

        # Track updates
        updated_count = 0

        # Update FAISS metadata
        try:
            with open(faiss_metadata_path, 'r', encoding='utf-8') as f:
                faiss_metadata = json.load(f)

            # Update the metadata recursively
            updated_faiss_metadata = update_recursively(faiss_metadata)

            # Count updates
            def count_differences(old_obj, new_obj):
                if isinstance(old_obj, dict) and isinstance(new_obj, dict):
                    return sum(count_differences(old_obj.get(k), new_obj.get(k))
                               for k in set(old_obj) | set(new_obj))
                elif isinstance(old_obj, list) and isinstance(new_obj, list):
                    return sum(count_differences(o, n) for o, n in zip(old_obj, new_obj))
                elif isinstance(old_obj, str) and isinstance(new_obj, str):
                    return 1 if old_obj != new_obj else 0
                return 0

            updated_count += count_differences(faiss_metadata, updated_faiss_metadata)

            # Save updated metadata
            with open(faiss_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(updated_faiss_metadata, f, indent=2, ensure_ascii=False)
            logger.info("Updated FAISS metadata")

        except Exception as e:
            logger.error(f"Error updating FAISS metadata: {e}")
            return False, f"Failed to update FAISS metadata: {e}"

        # Update image metadata
        try:
            with open(image_metadata_path, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)

            updated_image_metadata = update_recursively(image_metadata)
            updated_count += count_differences(image_metadata, updated_image_metadata)

            with open(image_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(updated_image_metadata, f, indent=2, ensure_ascii=False)
            logger.info("Updated image metadata")

        except Exception as e:
            logger.error(f"Error updating image metadata: {e}")
            return False, f"Failed to update image metadata: {e}"

        # Update processed_files.json
        try:
            if processed_files_path.exists():
                with open(processed_files_path, 'r', encoding='utf-8') as f:
                    processed_files = json.load(f)

                updated_processed_files = update_recursively(processed_files)
                updated_count += count_differences(processed_files, updated_processed_files)

                with open(processed_files_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_processed_files, f, indent=2)
                logger.info("Updated processed_files.json")

        except Exception as e:
            logger.error(f"Error updating processed_files.json: {e}")
            return False, f"Failed to update processed files list: {e}"

        # Rename the actual folder
        try:
            old_path.rename(new_path)
            logger.info(f"Renamed folder from {old_path} to {new_path}")
        except Exception as e:
            logger.error(f"Error renaming folder: {e}")
            return False, f"Failed to rename folder: {e}"

        return True, f"Folder renamed successfully. Updated {updated_count} paths across all metadata files."

    except Exception as e:
        logger.error(f"Error during folder rename: {e}")
        return False, f"Failed to rename folder: {e}"


def validate_folder_name(name: str) -> tuple[bool, str]:
    """
    Validates a folder name.

    Args:
        name: Folder name to validate

    Returns:
        Tuple of (valid: bool, message: str)
    """
    # Check for empty or whitespace
    if not name or not name.strip():
        return False, "Folder name cannot be empty"

    # Check length
    if len(name) > 255:
        return False, "Folder name is too long"

    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*'
    found_chars = [c for c in invalid_chars if c in name]
    if found_chars:
        return False, f"Folder name contains invalid characters: {', '.join(found_chars)}"

    # Check for reserved names (Windows)
    reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                      'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
                      'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    if name.upper() in reserved_names:
        return False, "This name is reserved by the system"

    return True, "Valid folder name"


def create_folder(parent_path: Path, folder_name: str) -> tuple[bool, str]:
    """
    Creates a new folder.

    Args:
        parent_path: Parent directory path
        folder_name: Name of the new folder

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate folder name
        valid, message = validate_folder_name(folder_name)
        if not valid:
            return False, message

        # Create full path
        new_folder_path = parent_path / folder_name

        # Check if folder already exists
        if new_folder_path.exists():
            return False, "Folder already exists"

        # Create folder
        new_folder_path.mkdir(parents=True, exist_ok=False)
        return True, "Folder created successfully"

    except PermissionError:
        return False, "Permission denied"
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        return False, f"Failed to create folder: {str(e)}"