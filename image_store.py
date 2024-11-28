from pathlib import Path
from PIL import Image
import json
import hashlib
import base64
from io import BytesIO
from typing import Tuple, List, Dict, Optional
from functools import lru_cache


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