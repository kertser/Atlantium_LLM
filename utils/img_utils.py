import base64
import hashlib
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any, Generator
import logging
import pymupdf
from contextlib import contextmanager
from utils.LLM_utils import encode_with_clip

import imagehash
import torch
from PIL import Image

from config import CONFIG
logger = logging.getLogger(__name__)

pymupdf.TOOLS.mupdf_display_errors(False)

class ImageProcessor:
    """Base class for image processing operations."""

    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """Convert an image to RGB format, preserving original colors."""
        if image.mode == 'RGB':
            return image

        if image.mode in ('RGBA', 'LA'):
            # Handle transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            if 'A' in image.mode:
                # Preserve original colors when removing transparency
                rgb_image = image.convert('RGB')
                if image.mode == 'RGBA':
                    background.paste(rgb_image, mask=image.split()[3])
                else:
                    background.paste(rgb_image)
                return background

        # Direct conversion for other modes
        return image.convert('RGB')

    @staticmethod
    def resize_image(image: Image.Image, max_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Resize the image if its dimensions exceed the maximum size while maintaining the aspect ratio.

        Args:
            image (Image.Image): The image to resize.
            max_size (Tuple[int, int]): Maximum allowable dimensions (width, height). Default is (512, 512).

        Returns:
            Image.Image: The resized image or the original if no resizing is needed.
        """
        if image.width <= max_size[0] and image.height <= max_size[1]:
            return image  # No resizing needed

        # Resize image while maintaining aspect ratio
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def calculate_hash(image: Union[Image.Image, str, bytes]) -> Optional[str]:
        """Calculate perceptual hash for various image formats."""
        try:
            if isinstance(image, str):
                try:
                    image_bytes = base64.b64decode(image)
                    image = Image.open(BytesIO(image_bytes))
                except Exception:
                    image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif not isinstance(image, Image.Image):
                raise ValueError("Invalid image format")

            image = ImageProcessor.convert_to_rgb(image)
            avg_hash = str(imagehash.average_hash(image))
            dhash = str(imagehash.dhash(image))
            phash = str(imagehash.phash(image))
            return f"{avg_hash}_{dhash}_{phash}"

        except Exception as e:
            logging.error(f"Error calculating image hash: {e}")
            return None

    @staticmethod
    def validate_image_data(image: Image.Image) -> bool:
        """Validate image data integrity."""
        try:
            # Basic size check
            if image.width < CONFIG.MIN_IMAGE_SIZE or image.height < CONFIG.MIN_IMAGE_SIZE:
                return False

            # Try to access image data
            try:
                image.load()
                _ = image.getdata()[0]  # Check first pixel
                return True
            except Exception:
                return False

        except Exception:
            return False

    @staticmethod
    async def calculate_image_similarity(
            image: Image.Image,
            query_text: str,
            distance: float,
            model: Any,
            device: str
    ) -> Dict:
        """
        Calculate image-text similarity using CLIP embeddings.

        Args:
            image: PIL Image to compare
            query_text: Text to compare against
            distance: Distance metric from previous calculations
            model: CLIP model instance
            device: Device to run calculations on ('cuda' or 'cpu')

        Returns:
            Dictionary containing similarity scores
        """
        try:
            # Ensure image is in RGB format
            if image.mode != "RGB":
                image = ImageProcessor.convert_to_rgb(image)

            with torch.no_grad():
                # Get image embeddings using encode_with_clip
                _, image_embeddings = encode_with_clip(
                    texts=None,
                    images=[image],
                    model=model,
                    device=device
                )
                if image_embeddings is None:
                    raise RuntimeError("Failed to generate image embeddings")

                # Get text embeddings
                text_embeddings, _ = encode_with_clip(
                    texts=[query_text],
                    images=None,
                    model=model,
                    device=device
                )
                if text_embeddings is None:
                    raise RuntimeError("Failed to generate text embeddings")

                # Convert numpy arrays to tensors
                image_embedding = torch.from_numpy(image_embeddings).to(device)
                query_embedding = torch.from_numpy(text_embeddings).to(device)

                # Calculate similarities
                semantic_similarity = (query_embedding @ image_embedding.T).item()
                content_similarity = (semantic_similarity + (1 - distance / 2)) / 2

                logging.debug(
                    f"Calculated similarities - semantic: {semantic_similarity:.4f}, content: {content_similarity:.4f}")

                return {
                    'similarity': content_similarity,
                    'semantic_similarity': semantic_similarity
                }

        except Exception as e:
            logging.error(f"Error calculating image similarity: {e}")
            logging.error("Full traceback:", exc_info=True)
            return {'similarity': 0, 'semantic_similarity': 0}

    @staticmethod
    def handle_pdf_image(image_bytes: bytes, page: Any) -> Optional[Image.Image]:
        """Handle problematic PDF images with multiple fallback methods."""
        try:
            # Try direct conversion first
            try:
                return Image.open(BytesIO(image_bytes))
            except Exception as e:
                logging.debug(f"Direct conversion failed: {e}")

            # Try PyMuPDF alternative extraction
            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
                return Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),  # Use tuple instead of list
                    pix.samples
                )
            except Exception as e:
                logging.debug(f"PyMuPDF conversion failed: {e}")

            return None
        except Exception as e:
            logging.error(f"Error handling PDF image: {e}")
            return None

    @staticmethod
    def compare_hashes(
            hash1: str,
            hash2: str,
            threshold: float = CONFIG.DEDUPLICATION_THRESHOLD
    ) -> bool:
        """Compare image hashes using weighted similarity."""
        try:
            avg1, dhash1, phash1 = hash1.split('_')
            avg2, dhash2, phash2 = hash2.split('_')

            avg_diff = imagehash.hex_to_hash(avg1) - imagehash.hex_to_hash(avg2)
            dhash_diff = imagehash.hex_to_hash(dhash1) - imagehash.hex_to_hash(dhash2)
            phash_diff = imagehash.hex_to_hash(phash1) - imagehash.hex_to_hash(phash2)

            # Calculate similarities (0-1 range)
            avg_sim = 1 - (avg_diff / 64)
            dhash_sim = 1 - (dhash_diff / 64)
            phash_sim = 1 - (phash_diff / 64)

            # Weighted average favoring perceptual hash
            similarity = (0.2 * avg_sim + 0.3 * dhash_sim + 0.5 * phash_sim)
            return similarity >= threshold

        except Exception as e:
            logging.error(f"Error comparing hashes: {e}")
            return False


class ImageStore(ImageProcessor):
    """Handles image storage, retrieval, and deduplication."""

    def __init__(self):
        """Initialize image store with configured paths."""
        super().__init__()
        self.base_path = CONFIG.STORED_IMAGES_PATH
        self.metadata_path = CONFIG.IMAGE_METADATA_PATH

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()
        self._verify_stored_images()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save current metadata to disk."""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
            raise

    def _verify_stored_images(self):
        """Verify all metadata entries reference valid images."""
        to_remove = []
        for image_id, data in self.metadata.items():
            try:
                image_path = Path(data["path"])
                if not image_path.exists():
                    logging.warning(f"Missing image: {image_id}")
                    to_remove.append(image_id)
                    continue

                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as e:
                    logging.error(f"Invalid image: {image_id}: {e}")
                    to_remove.append(image_id)

            except Exception as e:
                logging.error(f"Verification error: {image_id}: {e}")
                to_remove.append(image_id)

        for image_id in to_remove:
            del self.metadata[image_id]

        if to_remove:
            self._save_metadata()
            logging.info(f"Removed {len(to_remove)} invalid entries")

    @staticmethod
    def _generate_id(image: Image.Image, source: str, page: int) -> str:
        """Generate unique image ID based on content and source."""
        try:
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            content = buffer.getvalue()

            hasher = hashlib.sha256()
            hasher.update(content)
            hasher.update(str(source).encode())
            hasher.update(str(page).encode())

            return hasher.hexdigest()[:16]
        except Exception as e:
            logging.error(f"Error generating ID: {e}")
            raise

    def store_image(
            self,
            image: Image.Image,
            source_doc: str,
            page_num: int,
            caption: Optional[str] = None,
            context: Optional[str] = None
    ) -> str:
        """Store an image and return its ID."""
        try:
            image_id = self._generate_id(image, source_doc, page_num)
            original_mode = image.mode

            # Only convert if absolutely necessary
            if original_mode not in ('RGB', 'RGBA'):
                if original_mode in ('L', 'LA'):
                    # Keep grayscale as grayscale
                    if original_mode == 'LA':
                        background = Image.new('L', image.size, 255)
                        background.paste(image, mask=image.split()[1])
                        image = background
                else:
                    # Minimal conversion for other modes
                    image = image.convert('RGB')

            path = self.base_path / f"{image_id}.{CONFIG.PREFERRED_SAVE_FORMAT.lower()}"

            # Save with minimal processing
            save_params = {
                'format': CONFIG.PREFERRED_SAVE_FORMAT,
                'quality': 100,  # Maximum quality
            }

            if CONFIG.PREFERRED_SAVE_FORMAT == 'PNG':
                save_params.update({
                    'optimize': False,
                    'compress_level': 0  # No compression
                })

            image.save(path, **save_params)

            self.metadata[image_id] = {
                "source_document": str(source_doc),
                "page_number": page_num,
                "path": str(path.relative_to(CONFIG.BASE_DIR)),
                "caption": caption,
                "context": context,
                "width": image.width,
                "height": image.height,
                "original_mode": original_mode
            }

            self._save_metadata()
            return image_id

        except Exception as e:
            logging.error(f"Error storing image: {e}")
            raise

    def get_image(self, image_id: str) -> Tuple[Optional[Image.Image], Optional[Dict]]:
        """Retrieve image and metadata by ID."""
        try:
            if image_id not in self.metadata:
                return None, None

            metadata = self.metadata[image_id]
            try:
                path = CONFIG.BASE_DIR / metadata["path"]
            except Exception as e:
                logging.error(f"Invalid path in metadata for image {image_id}: {e}")
                return None, None

            if not path.exists():
                logging.error(f"Image not found: {path}")
                return None, None

            try:
                return Image.open(path), metadata
            except Exception as e:
                logging.error(f"Failed to open image {path}: {e}")
                return None, None

        except Exception as e:
            logging.error(f"Error retrieving image {image_id}: {e}")
            return None, None

    @contextmanager
    def open_image_safely(self, path: Union[str, Path]) -> Generator[Optional[Image.Image], None, None]:
        """Safely open and handle image cleanup."""
        img = None
        try:
            img = Image.open(path)
            yield img
        except Exception as e:
            logging.error(f"Error opening image {path}: {e}")
            yield None
        finally:
            if img:
                try:
                    img.close()
                except Exception:
                    pass

    def cleanup(self):
        """Clean up temporary resources."""
        try:
            # Clear the LRU cache for base64 encodings
            self.get_base64.cache_clear()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    @staticmethod
    def is_valid_format(image: Image.Image) -> bool:
        """Check if image format is supported."""
        return (
            image.format in CONFIG.SUPPORTED_IMAGE_FORMATS and
            image.mode in CONFIG.VALID_IMAGE_MODES
        )

    @lru_cache(maxsize=100)
    def get_base64(self, image_id: str) -> Optional[str]:
        """Get base64 encoded image data."""
        try:
            image, _ = self.get_image(image_id)
            if image is None:
                return None

            buffer = BytesIO()
            image.save(
                buffer,
                format=CONFIG.PREFERRED_SAVE_FORMAT,
                quality=CONFIG.IMAGE_QUALITY
            )
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            return None

    def get_document_images(self, source_doc: str) -> List[Dict]:
        """Get all images from a document."""
        try:
            return [
                {"id": img_id, **metadata}
                for img_id, metadata in self.metadata.items()
                if metadata["source_document"] == str(source_doc)
            ]
        except Exception as e:
            logging.error(f"Error getting document images: {e}")
            return []

    def delete_image(self, image_id: str) -> bool:
        """Delete image and its metadata."""
        try:
            if image_id not in self.metadata:
                return False

            path = Path(self.metadata[image_id]["path"])
            if path.exists():
                path.unlink()

            del self.metadata[image_id]
            self._save_metadata()
            return True

        except Exception as e:
            logging.error(f"Error deleting image: {e}")
            return False

    @staticmethod
    def merge_metadata(primary: Dict, secondary: Dict) -> Dict:
        """Merge metadata from two images."""
        merged = primary.copy()

        # Use 'source_document' instead of 'source' for consistency
        for field in ['caption', 'context', 'source_document']:
            if secondary.get(field):
                values = set(primary.get(field, '').split(' | '))
                values.add(secondary[field])
                merged[field] = ' | '.join(filter(None, values))

        return merged


class ImageClassifier(ImageProcessor):
    """Handles zero-shot image classification using CLIP."""

    def __init__(self, model=None, device='cuda'):
        """Initialize with CLIP model."""
        super().__init__()
        self.model = model
        self.device = device

        if self.model and str(self.model.device) != device:
            self.model = self.model.to(device)

    def cleanup(self):
        """Clean up model resources."""
        try:
            if self.model is not None:
                self.model.cpu()
                del self.model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error during classifier cleanup: {e}")

    def classify(
            self,
            image: Union[Image.Image, str],
            labels: List[str],
    ) -> Tuple[str, float]:
        """Perform zero-shot classification."""
        if not self.model:
            raise ValueError("Model required")

        try:
            # Process image
            if isinstance(image, str):
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Invalid image format")

            image = self.convert_to_rgb(image)

            # Process text: truncate and clean labels
            processed_labels = []
            for label in labels:
                cleaned_text = ' '.join(label.split())  # Remove extra whitespace
                truncated_text = cleaned_text[:100]  # Truncate to first 100 chars
                processed_labels.append(truncated_text)

            with torch.no_grad():
                # Get image embeddings using encode_with_clip
                _, image_embeddings = encode_with_clip(
                    texts=None,
                    images=[image],
                    model=self.model,
                    device=self.device
                )

                # Get text embeddings for labels
                text_embeddings, _ = encode_with_clip(
                    texts=processed_labels,
                    images=None,
                    model=self.model,
                    device=self.device
                )

                if image_embeddings is None or text_embeddings is None:
                    raise RuntimeError("Failed to generate embeddings")

                # Convert to tensors
                image_embeddings = torch.from_numpy(image_embeddings).to(self.device)
                text_embeddings = torch.from_numpy(text_embeddings).to(self.device)

                # Calculate similarity scores
                similarity = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
                probs = similarity[0].cpu().numpy()

                idx = probs.argmax().item()
                return processed_labels[idx], float(probs[idx])

        except Exception as e:
            logging.error(f"Classification error: {e}", exc_info=True)
            return "classification error", 0.0

    def rank_images_by_relevance(
            self,
            image_store: 'ImageStore',
            images: List[Dict],
            query_text: str,
            max_images: int = 12
    ) -> List[Dict]:
        """Rank images by their relevance to the query text."""
        if not self.model:
            raise ValueError("Model required")

        # Clean and prepare query text
        clean_query = ' '.join(query_text.split())
        text_segments = [clean_query[i:i + 100] for i in range(0, len(clean_query), 100)][:3]

        # Get text embeddings for all segments at once
        with torch.no_grad():
            text_embeddings, _ = encode_with_clip(
                texts=text_segments,
                images=None,
                model=self.model,
                device=self.device
            )
            if text_embeddings is None:
                raise RuntimeError("Failed to generate text embeddings")
            text_embeddings = torch.from_numpy(text_embeddings).to(self.device)

        ranked_images = []

        for img_info in images:
            try:
                image, metadata = image_store.get_image(img_info['image_id'])
                if image is None:
                    continue

                # Get image embeddings
                with torch.no_grad():
                    _, image_embeddings = encode_with_clip(
                        texts=None,
                        images=[image],
                        model=self.model,
                        device=self.device
                    )
                    if image_embeddings is None:
                        continue

                    image_embeddings = torch.from_numpy(image_embeddings).to(self.device)

                    # Calculate similarity with all text segments at once
                    similarities = (100.0 * image_embeddings @ text_embeddings.T)[0]
                    avg_score = similarities.mean().item()

                    img_info['relevance_score'] = avg_score
                    ranked_images.append(img_info)

            except Exception as e:
                logging.error(f"Error ranking image {img_info.get('image_id')}: {e}")
                continue

        # Sort by relevance score and limit
        ranked_images.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return ranked_images[:max_images]

    def deduplicate(
            self,
            images: List[Dict],
            similarity_threshold: float = CONFIG.DEDUPLICATION_THRESHOLD
    ) -> List[Dict]:
        """
        Deduplicate a list of images based on perceptual hash comparison.
        """
        if not images:
            return []

        unique_images = []

        for img in images:
            try:
                if 'image' not in img:
                    continue

                current_image = Image.open(BytesIO(base64.b64decode(img['image'])))

                # Use configured constant for size check
                if current_image.size[0] < CONFIG.MIN_ICON_SIZE or current_image.size[1] < CONFIG.MIN_ICON_SIZE:
                    continue

                # Calculate hash for the current image
                current_hash = self.calculate_hash(current_image)
                if not current_hash:
                    continue

                # Check similarity with already accepted unique images
                is_duplicate = False
                for existing in unique_images:
                    existing_hash = self.calculate_hash(
                        base64.b64decode(existing['image'])
                    )
                    if existing_hash and self.compare_hashes(
                            current_hash,
                            existing_hash,
                            threshold=similarity_threshold
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_images.append(img)
                    logging.info(f"Added unique image {img.get('image_id', 'unknown')}")

            except Exception as e:
                logging.error(f"Error processing image: {e}", exc_info=True)
                continue

        logging.info(f"Deduplicated {len(images)} images to {len(unique_images)} unique images")
        return unique_images
