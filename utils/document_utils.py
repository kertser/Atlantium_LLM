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

        # Track indices to remove, images, and chunks
        indices_to_remove = []
        images_to_remove = set()
        chunks_to_remove = set()

        # Get relative path for comparison
        try:
            relative_path = doc_path.relative_to(CONFIG.RAW_DOCUMENTS_PATH)
        except ValueError:
            relative_path = doc_path

        logger.info(f"Searching for entries related to {relative_path}")

        # Find all entries related to this document
        for idx, entry in enumerate(metadata):
            # Compare using the new path field
            entry_path = Path(entry.get('path', ''))

            # Check if paths match or if filenames match
            if (entry_path == relative_path or
                    entry_path.name == doc_path.name):
                logger.info(f"Found matching entry: {entry_path}")

                if idx not in indices_to_remove:
                    indices_to_remove.append(idx)

                # Collect image IDs
                if entry.get('type') == 'image':
                    image_data = entry.get('image', {})
                    if isinstance(image_data, dict):
                        image_id = image_data.get('id')
                        if image_id:
                            images_to_remove.add(image_id)

                # Collect chunk paths
                elif entry.get('type') == 'text-chunk':
                    chunk_path = entry.get('chunk')
                    if chunk_path:
                        chunks_to_remove.add(chunk_path)

        if not indices_to_remove:
            logger.info(f"Document {doc_path.name} not found in RAG system")
            return True, "Document not found in RAG system"

        logger.info(f"Found {len(indices_to_remove)} entries to remove")

        # Remove images
        if images_to_remove:
            logger.info(f"Found {len(images_to_remove)} images to remove")
            for image_id in images_to_remove:
                try:
                    if not image_store.delete_image(image_id):
                        logger.warning(f"Image ID not found or already deleted: {image_id}")
                    else:
                        logger.info(f"Deleted image {image_id}")
                except Exception as e:
                    logger.error(f"Error deleting image {image_id}: {e}")

        # Remove text chunks
        if chunks_to_remove:
            logger.info(f"Found {len(chunks_to_remove)} text chunks to remove")
            for chunk_path in chunks_to_remove:
                try:
                    full_path = CONFIG.RAG_DATA / chunk_path
                    if full_path.exists():
                        full_path.unlink()
                        logger.info(f"Deleted chunk file: {chunk_path}")
                except Exception as e:
                    logger.error(f"Error deleting chunk file {chunk_path}: {e}")

            # Clean up empty chunk directories
            chunk_dir = CONFIG.STORED_TEXT_CHUNKS_PATH / relative_path.stem
            if chunk_dir.exists() and not any(chunk_dir.iterdir()):
                try:
                    chunk_dir.rmdir()
                    logger.info(f"Removed empty chunk directory: {chunk_dir}")
                except Exception as e:
                    logger.error(f"Error removing chunk directory: {e}")

        try:
            # Create lists for keeping valid entries and their vectors
            valid_indices = [i for i in range(len(metadata)) if i not in indices_to_remove]
            new_metadata = [metadata[i] for i in valid_indices]

            # Create new index and add valid vectors
            new_index = faiss.IndexFlatL2(index.d)
            if valid_indices:
                vectors = []
                for idx in valid_indices:
                    try:
                        vector = faiss.vector_float_to_array(index.reconstruct(idx))
                        vectors.append(vector)
                    except RuntimeError as e:
                        logger.error(f"Error reconstructing vector at index {idx}: {e}")
                        continue

                if vectors:
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

            return True, f"Document removal completed: {len(indices_to_remove)} entries, {len(images_to_remove)} images, {len(chunks_to_remove)} chunks"

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