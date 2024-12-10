import logging
import base64
from io import BytesIO
from typing import Tuple, List, Dict, Union
from PIL import Image
import imagehash
import torch


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


def are_images_similar(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """
    Compare image hashes to determine similarity.
    Lower threshold means stricter matching.
    """
    try:
        # Split combined hashes
        avg1, dhash1, phash1 = hash1.split('_')
        avg2, dhash2, phash2 = hash2.split('_')

        # Convert string hashes back to imagehash objects
        avg_diff = imagehash.hex_to_hash(avg1) - imagehash.hex_to_hash(avg2)
        dhash_diff = imagehash.hex_to_hash(dhash1) - imagehash.hex_to_hash(dhash2)
        phash_diff = imagehash.hex_to_hash(phash1) - imagehash.hex_to_hash(phash2)

        # Images are similar if any two hash types indicate similarity
        similarities = [diff <= threshold for diff in (avg_diff, dhash_diff, phash_diff)]
        return sum(similarities) >= 2
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


def deduplicate_images(images: List[Dict], max_images: int = 8) -> List[Dict]:
    """
    Deduplicate images using perceptual hashing with size normalization.

    Args:
        images: List of image dictionaries
        max_images: Maximum number of images to return

    Returns:
        List of deduplicated images, limited to max_images
    """
    # Dictionary to track unique images by their perceptual hash
    unique_images = {}
    hash_groups = {}  # Track groups of similar images

    for img in images:
        image_id = img.get('image_id')
        if not image_id or not img.get('image'):
            continue

        # Calculate normalized hash and get original size
        image_hash, original_size = normalize_and_hash_image(img['image'])
        if not image_hash:
            continue

        # Check if this image is similar to any existing ones
        found_match = False
        for existing_hash in list(unique_images.keys()):
            if are_images_similar(image_hash, existing_hash):
                found_match = True
                # Use existing hash as the group key
                group_hash = existing_hash
                if img.get('similarity', 0) > unique_images[group_hash].get('similarity', 0):
                    # Keep metadata from old image
                    old_metadata = unique_images[group_hash]
                    unique_images[group_hash] = img
                    # Merge metadata
                    img = merge_image_metadata(img, old_metadata)

                # Track image ID in hash group
                if group_hash not in hash_groups:
                    hash_groups[group_hash] = set()
                hash_groups[group_hash].add(image_id)

                logging.info(f"Found similar images. ID: {image_id}, Size: {original_size}")
                break

        if not found_match:
            unique_images[image_hash] = img
            hash_groups[image_hash] = {image_id}
            logging.info(f"New unique image. ID: {image_id}, Size: {original_size}")

    # Log duplicate groups
    for hash_val, ids in hash_groups.items():
        if len(ids) > 1:
            logging.info(f"Duplicate image group with {len(ids)} variants: {ids}")

    # Convert to list and sort by similarity
    deduplicated = list(unique_images.values())
    deduplicated.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    logging.info(f"Deduplicated {len(images)} images to {len(deduplicated)} unique images")
    return deduplicated[:max_images]
