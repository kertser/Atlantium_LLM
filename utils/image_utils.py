import base64
import logging
from io import BytesIO
from typing import Tuple, List, Dict, Union

import imagehash
import torch
from PIL import Image

from config import CONFIG


def zero_shot_classification(
        image: Union[Image.Image, str],
        labels: List[str],
        model,
        processor,
        device
) -> Tuple[str, float]:
    """
    Perform zero-shot image classification using CLIP model.

    Args:
        image: PIL Image object or path to image file
        labels: List of text labels for classification
        model: Loaded CLIP model
        processor: Loaded CLIP processor
        device: Device to run model on ('cuda' or 'cpu')

    Returns:
        Tuple[str, float]: (predicted_label, confidence_score)
    """
    if model is None or processor is None:
        raise ValueError("Model and processor must be preloaded and passed to the function.")

    try:
        # Ensure model is on correct device
        if model.device.type != device:
            model = model.to(device)

        # Validate and ensure image is PIL Image
        try:
            if not isinstance(image, Image.Image):
                if isinstance(image, str):
                    image = Image.open(image)
                else:
                    raise ValueError("Input must be PIL Image or path to image")

            # Convert to RGB if needed
            if isinstance(image, Image.Image) and image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            logging.error(f"Failed to process image: {e}")
            return "image processing error", 0.0

        # Process inputs using CLIP processor
        inputs = processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Get prediction
        predicted_index = probs.argmax().item()
        predicted_label = labels[predicted_index]
        confidence = probs[0, predicted_index].item()

        return predicted_label, confidence

    except Exception as e:
        logging.error(f"Error during zero-shot classification: {e}", exc_info=True)
        return "classification error", 0.0


def normalize_and_hash_image(image_data: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[str, Tuple[int, int]]:
    """
    Normalize image size and calculate perceptual hash.

    Args:
        image_data: Base64 encoded image data
        target_size: Size to normalize to before hashing

    Returns:
        Tuple of (hash_string, original_size)
    """
    try:
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        original_size = image.size

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize for consistent hashing
        normalized = image.resize(target_size, Image.Resampling.LANCZOS)

        # Calculate multiple hash types
        avg_hash = imagehash.average_hash(normalized)
        dhash = imagehash.dhash(normalized)
        phash = imagehash.phash(normalized)

        # Convert hashes to binary strings for more precise comparison
        hash_string = f"{avg_hash}_{dhash}_{phash}"

        return hash_string, original_size
    except Exception as e:
        logging.error(f"Error calculating image hash: {e}")
        return None, None


def are_images_similar(hash1: str, hash2: str, threshold: float = CONFIG.DEDUPLICATION_THRESHOLD) -> bool:
    """
    Compare image hashes to determine similarity with structural comparison.
    Args:
        hash1: First image hash string
        hash2: Second image hash string
        threshold: Similarity between the images threshold (0.0 to 1.0)
    Returns:
        bool: True if images are more similar than the threshold
    """
    try:
        # Split combined hashes
        avg1, dhash1, phash1 = hash1.split('_')
        avg2, dhash2, phash2 = hash2.split('_')

        # Convert string hashes back to imagehash objects
        avg_diff = imagehash.hex_to_hash(avg1) - imagehash.hex_to_hash(avg2)
        dhash_diff = imagehash.hex_to_hash(dhash1) - imagehash.hex_to_hash(dhash2)
        phash_diff = imagehash.hex_to_hash(phash1) - imagehash.hex_to_hash(phash2)

        # Calculate similarity scores (0-1 range)
        avg_similarity = 1 - (avg_diff / 64)  # Hash size is 64 bits
        dhash_similarity = 1 - (dhash_diff / 64)
        phash_similarity = 1 - (phash_diff / 64)

        # Calculate weighted average (giving more weight to perceptual hash)
        similarity = (
                    0.2 * avg_similarity + 0.3 * dhash_similarity + 0.5 * phash_similarity)  # phash is best for structural similarity

        # logging.info(f"Image similarity score: {similarity:.4f}")
        return similarity >= threshold

    except Exception as e:
        logging.error(f"Error comparing image hashes: {e}")
        return False


def merge_image_metadata(primary_img: dict, secondary_img: dict) -> dict:
    """Merge metadata from two image records."""
    merged = primary_img.copy()

    # Merge captions
    if secondary_img.get('caption'):
        captions = set(primary_img.get('caption', '').split(' | '))
        captions.add(secondary_img['caption'])
        merged['caption'] = ' | '.join(filter(None, captions))

    # Merge contexts
    if secondary_img.get('context'):
        contexts = set(primary_img.get('context', '').split(' | '))
        contexts.add(secondary_img['context'])
        merged['context'] = ' | '.join(filter(None, contexts))

    # Merge sources
    if secondary_img.get('source'):
        sources = set(primary_img.get('source', '').split(' | '))
        sources.add(secondary_img['source'])
        merged['source'] = ' | '.join(filter(None, sources))

    return merged


def deduplicate_images(images: List[Dict], similarity_threshold: float = CONFIG.DEDUPLICATION_THRESHOLD) -> List[Dict]:
    """
    Deduplicate images with improved similarity detection
    """
    if not images:
        return []

    unique_images = []

    for img in images:
        try:
            # Skip if no image data
            if 'image' not in img:
                continue

            # Convert base64 to PIL Image for size check
            image_bytes = base64.b64decode(img['image'])
            current_image = Image.open(BytesIO(image_bytes))

            # Skip small images (likely logos/icons)
            if current_image.size[0] < 200 or current_image.size[1] < 200:
                # logging.info(f"Skipping small image: {current_image.size}")
                continue

            # Calculate hash for current image
            current_hash = normalize_and_hash_image(img['image'])[0]

            # Check similarity with existing unique images
            is_duplicate = False
            for existing in unique_images:
                existing_hash = normalize_and_hash_image(existing['image'])[0]
                if are_images_similar(current_hash, existing_hash, similarity_threshold):
                    is_duplicate = True
                    # logging.info("Found duplicate image")
                    break

            if not is_duplicate:
                unique_images.append(img)
                logging.info(f"Added unique image {img.get('image_id', 'unknown')}")

        except Exception as e:
            logging.error(f"Error processing image: {e}")
            continue

    logging.info(f"Deduplicated {len(images)} images to {len(unique_images)} unique images")
    return unique_images
