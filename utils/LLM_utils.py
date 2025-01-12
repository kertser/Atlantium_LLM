import logging
import time
from typing import Dict, Any

import torch
from fastapi import HTTPException
from openai import OpenAI
from transformers import AutoModel
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


def CLIP_init(model_name="jinaai/jina-clip-v2"):
    """
    Initialize Jina-CLIP model with detailed logging.
    """
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing CLIP on device: {device}")

        # Initialize model
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)

        if model is None:
            raise RuntimeError("Model initialization returned None")

        # Set model to evaluation mode
        model.eval()

        # Test encoding
        test_text = ["Test text"]
        with torch.no_grad():
            try:
                # Try encode_text first
                test_embedding = model.encode_text(test_text)
                if test_embedding is None:
                    raise RuntimeError("Test encoding returned None")
                # Convert to numpy if it's a tensor
                if isinstance(test_embedding, torch.Tensor):
                    test_embedding = test_embedding.cpu().numpy()
                logging.debug(f"Test encoding successful. Shape: {test_embedding.shape}")
            except AttributeError:
                # Fallback to get_text_features
                inputs = model.tokenizer(test_text, return_tensors="pt", padding=True).to(device)
                test_embedding = model.get_text_features(**inputs)
                if test_embedding is None:
                    raise RuntimeError("Test encoding returned None")
                test_embedding = test_embedding / test_embedding.norm(dim=-1, keepdim=True)
                test_embedding = test_embedding.cpu().numpy()
                logging.debug(f"Test encoding successful using get_text_features. Shape: {test_embedding.shape}")

            # Log the embedding dimension
            embedding_dim = test_embedding.shape[1]
            logging.info(f"Model produces embeddings of dimension: {embedding_dim}")
            if embedding_dim != CONFIG.EMBEDDING_DIMENSION:
                logging.warning(
                    f"Updating CONFIG.EMBEDDING_DIMENSION from {CONFIG.EMBEDDING_DIMENSION} to {embedding_dim}")
                CONFIG.EMBEDDING_DIMENSION = embedding_dim

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
