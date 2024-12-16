import os
import platform
import subprocess
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
import shutil
import faiss
import json
from config import CONFIG
from utils.FAISS_utils import load_faiss_index, load_metadata, save_faiss_index, save_metadata
from utils.image_store import ImageStore

logger = logging.getLogger(__name__)


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
    Removes a document and all its associated data from the RAG system.
    Args:
        doc_path: Path to the document to remove
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        logger.info(f"Starting removal of document: {doc_path}")

        # Load current index and metadata
        index = load_faiss_index(CONFIG.FAISS_INDEX_PATH)
        metadata = load_metadata(CONFIG.METADATA_PATH)
        image_store = ImageStore(CONFIG.STORED_IMAGES_PATH)

        logger.info(f"Loaded FAISS index with {len(metadata)} entries")

        # Track indices to remove and image IDs
        indices_to_remove = []
        images_to_remove = set()
        doc_path_str = str(doc_path)
        doc_name = doc_path.name  # Get just the filename

        logger.info(f"Searching for entries related to {doc_name}")

        # Find all entries and images related to this document
        for idx, entry in enumerate(metadata):
            # Check against full path and normalized paths
            entry_source = entry.get('source_file', '')
            entry_source_normalized = str(Path(entry_source)).replace('\\', '/')
            doc_path_normalized = str(doc_path).replace('\\', '/')

            # Check if paths match after normalization or if filenames match
            if (Path(entry_source_normalized).name == doc_name or
                    Path(doc_path_normalized).name == doc_name):
                logger.info(f"Found matching entry by filename: {entry_source}")
                if idx not in indices_to_remove:
                    indices_to_remove.append(idx)

                # If it's an image entry, collect image ID
                if entry.get('type') == 'image':
                    content = entry.get('content', {})
                    if isinstance(content, dict):
                        image_id = content.get('image_id')
                        if image_id:
                            images_to_remove.add(image_id)

            # Check relative path
            rel_path = entry.get('relative_path', '')
            if rel_path and Path(rel_path).name == doc_name:
                logger.info(f"Found matching entry by relative path: {rel_path}")
                if idx not in indices_to_remove:
                    indices_to_remove.append(idx)

        if not indices_to_remove:
            logger.info(f"Document {doc_name} not found in RAG system")
            return True, "Document not found in RAG system"

        logger.info(f"Found {len(indices_to_remove)} entries to remove")
        if images_to_remove:
            logger.info(f"Found {len(images_to_remove)} images to remove")

            # Remove all associated images
            for image_id in images_to_remove:
                try:
                    if not image_store.delete_image(image_id):
                        logger.warning(f"Image ID not found or already deleted: {image_id}")
                    else:
                        logger.info(f"Deleted image {image_id} associated with {doc_path.name}")
                except Exception as e:
                    logger.error(f"Error deleting image {image_id}: {e}")
                    # Continue with other images instead of failing the whole operation

        try:
            # Create lists for keeping valid entries and their vectors
            valid_indices = [i for i in range(len(metadata)) if i not in indices_to_remove]
            new_metadata = [metadata[i] for i in valid_indices]

            # Create new index and add valid vectors
            new_index = faiss.IndexFlatL2(index.d)
            if valid_indices:  # Only add vectors if we have valid indices
                vectors = []
                for idx in valid_indices:
                    try:
                        vector = faiss.vector_float_to_array(index.reconstruct(idx))
                        vectors.append(vector)
                    except RuntimeError as e:
                        logger.error(f"Error reconstructing vector at index {idx}: {e}")
                        continue

                if vectors:  # Only add if we have valid vectors
                    vectors_array = np.vstack(vectors)
                    new_index.add(vectors_array)

            # Verify index integrity
            if new_index.ntotal != len(new_metadata):
                logger.error(f"Index/metadata mismatch after rebuild: {new_index.ntotal} vs {len(new_metadata)}")
                return False, "Index and metadata are out of sync after removal"

            # Save updated index and metadata
            save_faiss_index(new_index, CONFIG.FAISS_INDEX_PATH)
            save_metadata(new_metadata, CONFIG.METADATA_PATH)
            logger.info(f"Saved updated FAISS index with {len(new_metadata)} entries")

            # Update processed files list
            update_processed_files_list(doc_path, remove=True)
            logger.info("Updated processed files list")

            return True, f"Document and {len(images_to_remove)} associated images removed successfully"

        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {str(e)}")
            return False, f"Failed to rebuild FAISS index: {str(e)}"

    except Exception as e:
        logger.error(f"Error removing document from RAG: {str(e)}")
        return False, f"Failed to remove document from RAG: {str(e)}"


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
    Renames a folder and updates all RAG references.

    Args:
        old_path: Current folder path
        new_path: New folder path

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Load current metadata
        metadata = load_metadata(CONFIG.METADATA_PATH)

        # Update paths in metadata
        old_path_str = str(old_path)
        new_path_str = str(new_path)

        for entry in metadata:
            # Update source_file
            source_file = entry.get('source_file', '')
            if source_file.startswith(old_path_str):
                entry['source_file'] = source_file.replace(old_path_str, new_path_str)

            # Update relative_path
            rel_path = entry.get('relative_path', '')
            if rel_path.startswith(str(old_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH))):
                entry['relative_path'] = rel_path.replace(
                    str(old_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH)),
                    str(new_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH))
                )

            # Update directory
            directory = entry.get('directory', '')
            if directory.startswith(old_path_str):
                entry['directory'] = directory.replace(old_path_str, new_path_str)

            # Update image paths if present
            if entry.get('type') == 'image' and isinstance(entry.get('content'), dict):
                content = entry['content']
                source_doc = content.get('source_doc', '')
                if source_doc.startswith(old_path_str):
                    content['source_doc'] = source_doc.replace(old_path_str, new_path_str)

        # Save updated metadata
        save_metadata(metadata, CONFIG.METADATA_PATH)

        # Update processed files list
        with open("processed_files.json", 'r') as f:
            processed_files = json.load(f)

        updated_files = []
        for file_path in processed_files:
            if file_path.startswith(old_path_str):
                updated_files.append(file_path.replace(old_path_str, new_path_str))
            else:
                updated_files.append(file_path)

        with open("processed_files.json", 'w') as f:
            json.dump(updated_files, f, indent=2)

        # Rename the actual folder
        old_path.rename(new_path)

        return True, "Folder renamed successfully"

    except Exception as e:
        logger.error(f"Error renaming folder: {str(e)}")
        return False, f"Failed to rename folder: {str(e)}"


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