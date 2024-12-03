import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import requests
import logging
import time


def openai_post_request(messages, model_name, max_tokens, temperature, api_key):
    """Send request to OpenAI API with rate limit handling"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    max_retries = 5
    base_delay = 1  # Start with 1 second delay

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30.0)

            if response.status_code == 429:  # Rate limit exceeded
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Rate limit reached. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to get response from OpenAI API after multiple retries"
                )
            logging.error(f"Error during OpenAI request (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(base_delay * (2 ** attempt))

    raise HTTPException(status_code=500, detail="Maximum retries reached for OpenAI API request")

def CLIP_init(model_name = "openai/clip-vit-base-patch32"):
    try:
        # Set device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Initialize CLIP model and processor
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_name)

        # Validate initialization
        assert clip_model is not None, "CLIP model failed to load."
        assert clip_processor is not None, "CLIP processor failed to initialize."

        # Set model to evaluation mode
        clip_model.eval()
        print(f"CLIP model {model_name} initialized on {device}.")

        return clip_model, clip_processor, device

    except Exception as e:
        print(f"Error initializing CLIP model: {str(e)}")

def process_image_for_clip(image):
    """
    Process an image to ensure it's in the correct format for CLIP
    """
    try:
        # Convert PIL Image to RGB if it isn't already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def encode_with_clip(texts, images, model, processor, device):
    """
    Encodes text and images using CLIP.
    :param texts: List of strings to encode.
    :param images: List of PIL Image objects to encode.
    :return: Tuple of (text_embeddings, image_embeddings) as NumPy arrays.
    """
    text_embeddings = []
    image_embeddings = []

    # Encode texts
    if texts and isinstance(texts, list):
        try:
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
            text_embeddings = text_features.cpu().detach().numpy()
        except Exception as e:
            print(f"Error encoding text: {e}")

    # Encode images
    for image in images:
        try:
            if image is None:
                continue
            # Ensure image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_input = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
            image_embeddings.append(image_features.cpu().detach().numpy())
        except Exception as e:
            print(f"Error encoding image: {e}")
            continue

    # Flatten image embeddings if needed
    if image_embeddings:
        try:
            image_embeddings = np.vstack(image_embeddings)
        except ValueError as e:
            print(f"Error flattening image embeddings: {e}")
            image_embeddings = np.array([])

    return text_embeddings, image_embeddings