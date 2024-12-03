from pathlib import Path
from PIL import Image
import json
import hashlib
import base64
from io import BytesIO
from typing import Tuple, List, Set, Dict, Optional
from functools import lru_cache
import logging
import imagehash


class ImageStore:
    def __init__(self, base_path: Path):
        """
        Initialize the ImageStore with a base path for storing images and metadata

        Args:
            base_path (Path): Base directory for storing images and metadata
        """
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.metadata_path = self.base_path / "image_metadata.json"

        # Create directories if they don't exist
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load or initialize image metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save image metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_image_id(self, image: Image.Image, source_doc: str, page_num: int) -> str:
        """Generate a unique ID for an image based on its content and source"""
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create hash from image content and metadata
        hasher = hashlib.sha256()
        hasher.update(img_byte_arr)
        hasher.update(source_doc.encode())
        hasher.update(str(page_num).encode())

        return hasher.hexdigest()[:16]

    def store_image(self, image: Image.Image, source_doc: str, page_num: int,
                    caption: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Store an image and return its ID

        Args:
            image (Image.Image): PIL Image to store
            source_doc (str): Source document path or identifier
            page_num (int): Page number where the image was found
            caption (Optional[str]): Optional caption for the image
            context (Optional[str]): Surrounding text context for the image

        Returns:
            str: Unique identifier for the stored image
        """
        image_id = self._generate_image_id(image, source_doc, page_num)

        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Save image
        image_path = self.images_path / f"{image_id}.png"
        image.save(image_path, "PNG")

        # Store metadata
        self.metadata[image_id] = {
            "source_document": source_doc,
            "page_number": page_num,
            "path": str(image_path),
            "caption": caption,
            "context": context,
            "width": image.width,
            "height": image.height
        }

        self._save_metadata()
        return image_id

    def get_image(self, image_id: str) -> Tuple[Optional[Image.Image], Optional[Dict]]:
        """
        Retrieve an image and its metadata by ID

        Args:
            image_id (str): Unique identifier for the image

        Returns:
            Tuple[Optional[Image.Image], Optional[Dict]]: Tuple of (image, metadata) or (None, None) if not found
        """
        if image_id not in self.metadata:
            return None, None

        image_data = self.metadata[image_id]
        try:
            image = Image.open(image_data["path"])
            return image, image_data
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return None, None

    @lru_cache(maxsize=100)
    def get_base64_image(self, image_id: str) -> Optional[str]:
        """
        Get base64 encoded image for web display

        Args:
            image_id (str): Unique identifier for the image

        Returns:
            Optional[str]: Base64 encoded image or None if not found
        """
        image, _ = self.get_image(image_id)
        if image is None:
            return None

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def get_images_from_document(self, source_doc: str) -> List[Dict]:
        """
        Get all images from a specific document

        Args:
            source_doc (str): Source document path or identifier

        Returns:
            List[Dict]: List of image metadata dictionaries
        """
        return [
            {"id": img_id, **metadata}
            for img_id, metadata in self.metadata.items()
            if metadata["source_document"] == source_doc
        ]

    def delete_image(self, image_id: str) -> bool:
        """
        Delete an image and its metadata

        Args:
            image_id (str): Unique identifier for the image

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if image_id not in self.metadata:
            return False

        try:
            # Remove image file
            image_path = Path(self.metadata[image_id]["path"])
            if image_path.exists():
                image_path.unlink()

            # Remove metadata
            del self.metadata[image_id]
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Error deleting image {image_id}: {e}")
            return False


def calculate_image_hash(image: Image.Image) -> str:
    """
    Calculate a perceptual hash of an image for deduplication.
    Uses average hash, difference hash, and perceptual hash for better accuracy.

    Args:
        image (Image.Image): PIL Image object
    Returns:
        str: Combined hash string
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Calculate multiple hash types for better accuracy
    avg_hash = str(imagehash.average_hash(image))
    dhash = str(imagehash.dhash(image))
    phash = str(imagehash.phash(image))

    # Combine hashes
    return f"{avg_hash}_{dhash}_{phash}"


def merge_image_contexts(contexts: List[Dict]) -> Dict:
    """Merge context dictionaries for duplicate images with smarter deduplication."""

    # Get shortest caption among duplicates
    captions = [ctx.get("caption", "") for ctx in contexts if ctx.get("caption")]
    caption = min(captions, key=len) if captions else ""

    # Deduplicate source documents
    source_doc = contexts[0].get("source_document", "") if contexts else ""

    # Get unique page numbers
    pages = {str(ctx.get("page_number")) for ctx in contexts if ctx.get("page_number")}
    page_numbers = "; ".join(sorted(pages))

    # Get non-empty context with shortest length
    contexts_list = [ctx.get("context", "") for ctx in contexts if ctx.get("context")]
    context = min(contexts_list, key=len) if contexts_list else ""

    return {
        "source_document": source_doc,
        "page_number": page_numbers,
        "caption": caption,
        "context": context
    }


def deduplicate_images(image_store: 'ImageStore') -> Tuple[Dict[str, Set[str]], Dict[str, Dict]]:
    """
    Find and group duplicate images in the image store.

    Args:
        image_store: ImageStore instance
    Returns:
        Tuple[Dict[str, Set[str]], Dict[str, Dict]]:
            - Dictionary mapping hash to set of image IDs
            - Dictionary mapping hash to merged context
    """
    hash_to_ids: Dict[str, Set[str]] = {}
    hash_to_context: Dict[str, List[Dict]] = {}

    logging.info("Starting image deduplication process...")

    for image_id, metadata in image_store.metadata.items():
        try:
            image_path = metadata.get("path")
            if not image_path:
                continue

            image = Image.open(image_path)
            image_hash = calculate_image_hash(image)

            if image_hash not in hash_to_ids:
                hash_to_ids[image_hash] = set()
                hash_to_context[image_hash] = []

            hash_to_ids[image_hash].add(image_id)
            hash_to_context[image_hash].append(metadata)

        except Exception as e:
            logging.error(f"Error processing image {image_id}: {e}")
            continue

    # Merge contexts for duplicate groups
    merged_contexts = {
        hash_val: merge_image_contexts(contexts)
        for hash_val, contexts in hash_to_context.items()
        if len(hash_to_ids[hash_val]) > 1
    }

    logging.info(f"Found {len([ids for ids in hash_to_ids.values() if len(ids) > 1])} groups of duplicate images")

    return hash_to_ids, merged_contexts


def remove_duplicate_images(image_store: 'ImageStore', hash_to_ids: Dict[str, Set[str]],
                            merged_contexts: Dict[str, Dict]) -> None:
    """
    Remove duplicate images and update metadata with merged contexts.

    Args:
        image_store: ImageStore instance
        hash_to_ids: Dictionary mapping hash to set of image IDs
        merged_contexts: Dictionary mapping hash to merged context
    """
    for image_hash, image_ids in hash_to_ids.items():
        if len(image_ids) > 1:
            # Keep the first ID, remove others
            keep_id = next(iter(image_ids))
            remove_ids = image_ids - {keep_id}

            # Update metadata for kept image
            if image_hash in merged_contexts:
                image_store.metadata[keep_id].update(merged_contexts[image_hash])

            # Remove duplicate images
            for remove_id in remove_ids:
                image_store.delete_image(remove_id)

    # Save updated metadata
    image_store._save_metadata()
    logging.info("Completed duplicate image removal and metadata update")


def update_faiss_metadata(metadata: List[Dict], hash_to_ids: Dict[str, Set[str]],
                          merged_contexts: Dict[str, Dict]) -> List[Dict]:
    """
    Update FAISS metadata to reflect image deduplication.

    Args:
        metadata: List of FAISS metadata entries
        hash_to_ids: Dictionary mapping hash to set of image IDs
        merged_contexts: Dictionary mapping hash to merged context
    Returns:
        List[Dict]: Updated metadata list
    """
    updated_metadata = []
    seen_hashes = set()

    for entry in metadata:
        if entry.get('type') != 'image':
            updated_metadata.append(entry)
            continue

        image_id = entry.get('image_id')
        if not image_id:
            continue

        # Find hash for this image ID
        matching_hash = None
        for hash_val, ids in hash_to_ids.items():
            if image_id in ids:
                matching_hash = hash_val
                break

        if matching_hash and matching_hash not in seen_hashes:
            # Update with merged context and keep
            keep_id = next(iter(hash_to_ids[matching_hash]))
            if matching_hash in merged_contexts:
                entry.update(merged_contexts[matching_hash])
            entry['image_id'] = keep_id
            updated_metadata.append(entry)
            seen_hashes.add(matching_hash)

    return updated_metadata
