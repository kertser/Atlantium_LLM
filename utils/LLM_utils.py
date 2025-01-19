import logging
import time
import os
from typing import Dict, Any
from unittest.mock import patch

from PIL import Image

import torch
from fastapi import HTTPException
from openai import OpenAI

from transformers import AutoModel, BlipProcessor, BlipForConditionalGeneration
from transformers.dynamic_module_utils import get_imports
from transformers import BeitImageProcessor, BeitForImageClassification
from config import CONFIG


def openai_post_request(messages: list, model_name: str, api_key: str, max_tokens: int = None,
                        temperature: float = None, functions: list = None, function_call: str = None) -> Dict[str, Any]:
    """Send request using OpenAI client library with rate limit handling"""
    client = OpenAI(api_key=api_key)
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # Format messages according to the new API format
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if not content:  # Skip empty messages
                    continue

                formatted_msg = {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": str(content)
                        }
                    ]
                }
                formatted_messages.append(formatted_msg)

            if not formatted_messages:  # Check if we have any messages
                raise ValueError("No valid messages to send")

            kwargs = {
                "model": model_name,
                "messages": formatted_messages,
                "response_format": {"type": "text"}
            }

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if functions is not None:
                kwargs["functions"] = functions
            if function_call is not None:
                kwargs["function_call"] = function_call

            response = client.chat.completions.create(**kwargs)

            # Return formatted response with function call if present
            message = response.choices[0].message
            response_dict = {
                'choices': [{
                    'message': {
                        'content': message.content if message.content else ""
                    }
                }]
            }

            if hasattr(message, 'function_call') and message.function_call:
                response_dict['choices'][0]['message']['function_call'] = {
                    'name': message.function_call.name,
                    'arguments': message.function_call.arguments
                }

            return response_dict

        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API error after {max_retries} retries: {str(e)}"
                )
            logging.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(base_delay * (2 ** attempt))


def gpt_vision_request(base64_image: str, api_key: str) -> str:
    """
    Simple synchronous function for getting image description using OpenAI Vision API.

    Args:
        base64_image: Base64 encoded image string
        api_key: OpenAI API key

    Returns:
        str: Description of the image or empty string if failed
    """
    try:
        client = OpenAI(api_key=api_key)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe what you see on this image."
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model=CONFIG.BASE_VISION_MODEL,
            messages=messages,
            max_tokens=50
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()

        return ""

    except Exception as err:
        logging.error(f"Vision request failed: {err}")
        return ""

def BLIP_init(device='cuda'):
    try:
        if device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA is not available. Using CPU for BLIP.")
            device = 'cpu'
        logging.info(f"Initializing Blip model on device: {device}")

        processor = BlipProcessor.from_pretrained(CONFIG.BLIP_MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(CONFIG.BLIP_MODEL_NAME).to(device)

        logging.info("Blip model initialized successfully")
        return processor, model

    except Exception as e:
        logging.error(f"Blip initialization failed with error: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        return None, None

def blip_vision_request(image: Image.Image, processor, model, device='cuda') -> str:
    try:
        inputs = processor(image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)

    except Exception:
        logging.error("Blip model failed:")
        return ""


def grok_post_request(messages, model_name="grok-beta", max_tokens=128, temperature=0, api_key=""):
    """Send request to Grok using OpenAI client library with rate limit handling"""
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    """
    messages=[
    {"role": "assistant", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
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
            logging.info(response.choices[0].message)
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


def CLIP_init(model_name="jinaai/jina-clip-v2", device_str: str = None):
    """
    Initialize Jina-CLIP model with detailed logging and enhanced functionality.

    Args:
        model_name (str): Name or path of the CLIP model
        device_str (str): Optional device specification (e.g., "cpu", "cuda")

    Returns:
        tuple: (model, device) or (None, None) if initialization fails
    """

    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        """Handle unnecessary flash_attn dependency"""
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports

    try:
        # Set device
        if device_str is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        logging.info(f"Initializing CLIP on device: {device}")

        # Set dtype based on device
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Set configuration for model initialization
        if device.type == "cpu":
            config = {
                "use_flash_attention": False,  # Disable flash attention
                "use_memory_efficient_attention": False,  # Disable memory efficient attention
                "enable_xformers": False  # Disable xformers
            }
        else:
            config = {}

        # Initialize model with flash_attn patch
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                config=config
            ).to(device)

        if model is None:
            raise RuntimeError("Model initialization returned None")

        # Set model to evaluation mode
        model.eval()

        logging.info("CLIP model initialized successfully")
        return model, device

    except Exception as e:
        logging.error(f"CLIP initialization failed with error: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        return None, None


def encode_with_clip(texts, images, model, device):
    """
    Encode texts and images using Jina-CLIP.
    """
    text_embeddings = None
    image_embeddings = None

    # Encode texts
    if texts and isinstance(texts, list):
        try:
            with torch.no_grad():
                try:
                    text_embeddings = model.encode_text(texts)
                except AttributeError:
                    inputs = model.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=CONFIG.CHUNK_SIZE
                    ).to(device)
                    text_features = model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_embeddings = text_features

                if isinstance(text_embeddings, torch.Tensor):
                    text_embeddings = text_embeddings.cpu().numpy()

                logging.debug(f"Generated text embeddings shape: {text_embeddings.shape}")

        except Exception as e:
            logging.error(f"Error encoding text: {str(e)}")
            logging.error("Text encoding traceback:", exc_info=True)
            raise

    # Encode images
    if images and isinstance(images, list):
        try:
            with torch.no_grad():
                try:
                    image_embeddings = model.encode_image(images)
                except AttributeError as e:
                    logging.error(f"Error encoding images: {str(e)}")

                if isinstance(image_embeddings, torch.Tensor):
                    image_embeddings = image_embeddings.cpu().numpy()

                logging.debug(f"Generated image embeddings shape: {image_embeddings.shape}")

        except Exception as e:
            logging.error(f"Error encoding images: {str(e)}")
            logging.error("Image encoding traceback:", exc_info=True)
            raise

    return text_embeddings, image_embeddings

def BEIT_init():
    # Load the feature extractor and model
    feature_extractor = BeitImageProcessor.from_pretrained('amaye15/Beit-Base-Image-Orientation-Fixer')
    model = BeitForImageClassification.from_pretrained('amaye15/Beit-Base-Image-Orientation-Fixer')

    return model, feature_extractor

def correct_image_orientation(image: Image.Image, model, feature_extractor) -> Image.Image:
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Define rotation angles based on model's class indices
    rotation_angles = {0: 0, 1: 90, 2: 180, 3: 270}
    rotation_angle = rotation_angles.get(predicted_class_idx, 0)

    # Rotate the image to correct orientation
    if rotation_angle != 0:
        image = image.rotate(-rotation_angle, expand=True)

    return image
