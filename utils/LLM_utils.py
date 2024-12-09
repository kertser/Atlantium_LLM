import torch
from fastapi import HTTPException
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import requests
import logging
import time
from openai import OpenAI


def openai_post_request(messages, model_name, max_tokens, temperature, api_key):
    """Send request using OpenAI client library with rate limit handling"""
    client = OpenAI(api_key=api_key)
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return {"choices": [{"message": {"content": response.choices[0].message.content}}]}

        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API error after {max_retries} retries: {str(e)}"
                )
            logging.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(base_delay * (2 ** attempt))

    raise HTTPException(status_code=500, detail="Maximum retries reached for OpenAI API request")


def grok_post_request(messages, model_name="grok-beta", max_tokens=128, temperature=0, api_key=""):
    """Send request to Grok using OpenAI client library with rate limit handling"""
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    """
    messages=[
    {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
    {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ]
    """

    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(response.choices[0].message)
            return {"choices": [{"message": {"content": response.choices[0].message.content}}]}

        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API error after {max_retries} retries: {str(e)}"
                )
            logging.error(f"Grok API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(base_delay * (2 ** attempt))

    raise HTTPException(status_code=500, detail="Maximum retries reached for OpenAI API request")

def CLIP_init(model_name="openai/clip-vit-base-patch32"):
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
    Returns: Tuple of (text_embeddings, image_embeddings) as NumPy arrays.
    """
    text_embeddings = []
    image_embeddings = []

    # Encode texts
    if texts and isinstance(texts, list):
        try:
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_embeddings = text_features.cpu().detach().numpy()
        except Exception as e:
            logging.error(f"Error encoding text: {e}")

    # Encode images
    if images and isinstance(images, list):
        try:
            processed_images = []
            for image in images:
                if image is None:
                    continue
                # Ensure image is in RGB mode
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_images.append(image)

            if processed_images:
                image_inputs = processor(images=processed_images, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                with torch.no_grad():
                    image_features = model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_embeddings = image_features.cpu().detach().numpy()

        except Exception as e:
            logging.error(f"Error encoding images: {e}")

    return text_embeddings, image_embeddings
