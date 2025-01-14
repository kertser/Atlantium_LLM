import torch
from transformers import AutoModel
from unittest.mock import patch
import os
from transformers.dynamic_module_utils import get_imports
import functools
import time


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


@timer
def initialize_clip(model_path: str, device: str = None) -> None:
    try:
        if device is None:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            # Convert string to torch.device
            device = torch.device(device)

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        print(f"Initializing CLIP on device: {device}")

        # Handle unnecessary flash_attn dependency for CPU
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(device)

        if model is None:
            raise RuntimeError("Model initialization returned None")

        # Set model to evaluation mode
        model.eval()

        # Test encoding
        test_text = ["Test text"]
        with torch.no_grad():
            if not hasattr(model, "encode_text"):
                raise RuntimeError("Model does not support 'encode_text' method")
            test_embedding = model.encode_text(test_text)
            if test_embedding is None:
                raise RuntimeError("Test encoding returned None")
            if isinstance(test_embedding, torch.Tensor):
                test_embedding = test_embedding.cpu().numpy()
            print(f"Test encoding successful. Shape: {test_embedding.shape}")

        print("CLIP model initialized successfully")

    except Exception as e:
        print(f"CLIP initialization failed with error: {str(e)}")


# Initialize the model
model_name = "jinaai/jina-clip-v2"
initialize_clip(model_name, device="cpu")
