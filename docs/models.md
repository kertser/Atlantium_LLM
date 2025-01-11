# Models Documentation

## Overview

The models package manages AI model integrations, prompt engineering, and response generation. It combines CLIP for multimodal embeddings with GPT models for context-aware responses.

## Core Components

### PromptLoader
Singleton class managing system prompts and templates.
```python
class PromptLoader:
    def get_system_prompt(self, key: str) -> str
    def get_instructions(self, instruction_type: str) -> List[str]
    def get_template(self, key: str) -> str
    def get_example(self, example_key: str) -> str
    def format_template(self, template_key: str, **kwargs) -> str
    def get_no_answer_prompt(self) -> str
    def get_conflict_resolution_prompt(self) -> str
    def get_ambiguity_handling_prompt(self) -> str
```

### PromptBuilder
Class for constructing various types of prompts.
```python
class PromptBuilder:
    def build_chat_prompt(
        self,
        query_text: str,
        contexts: List[str],
        images: List[Dict],
        chat_history: List[Dict],
        is_technical: bool = False
    ) -> str

    def build_messages(self, prompt: str) -> List[Dict[str, str]]
    def build_no_answer_message(self, query_text: str) -> List[Dict[str, str]]
    def build_conflict_resolution_message(self, conflicting_docs: List[Dict]) -> List[Dict[str, str]]
    def build_ambiguity_message(self, query_text: str) -> List[Dict[str, str]]
```

## Model Integrations

### CLIP Integration
Used for multimodal embeddings and image analysis.
```python
def CLIP_init(model_name: str = "openai/clip-vit-base-patch32"):
    """Initialize CLIP model and processor."""

def encode_with_clip(texts, images, model, processor, device):
    """Generate embeddings for text and images."""

def zero_shot_classification(image, labels, model, processor, device):
    """Perform zero-shot image classification."""
```

### GPT Integration
Handles language model interactions.
```python
def openai_post_request(
    messages: List[Dict],
    model_name: str,
    api_key: str,
    max_tokens: int = None,
    temperature: float = None
) -> Dict:
    """Send request to OpenAI API."""
```

## Prompt Templates

### System Prompts
Located in `templates/prompts.yaml`:
```yaml
system:
  technical_assistant: |
    # Core guidelines for technical responses
  vision_assistant: |
    # Guidelines for image analysis
```

### Response Templates
```yaml
templates:
  chat_prompt: |
    # Template for chat responses
  technical_response: |
    # Template for technical information
  image_query: |
    # Template for image analysis
```

### Error Handling
```yaml
error_handling:
  insufficient_data: |
    # Template for handling insufficient data
  conflict_handling: |
    # Template for resolving conflicts
  ambiguity_handling: |
    # Template for clarifying ambiguity
```

## Configuration

### Model Settings
From `config.py`:
```python
CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
EMBEDDING_DIMENSION: int = 512
USE_GPU: bool = True
GPT_MODEL: str = "gpt-4o-mini"
GPT_VISION_MODEL: str = "gpt-4o"
VISION_MAX_TOKENS: int = 4096
```

### Thresholds
```python
SIMILARITY_THRESHOLD: float = 0.75      # Text similarity
IMAGE_SIMILARITY_THRESHOLD: float = 0.35  # Image similarity
TECHNICAL_CONFIDENCE_THRESHOLD: float = 0.75
DEDUPLICATION_THRESHOLD: float = 0.70
```

## Response Formatting

### Technical Response Structure
1. Overview/Context
2. Technical Details
3. Specifications/Parameters
4. Application/Usage
5. Safety Considerations
6. References

### Vision Response Structure
1. Image Analysis
2. Technical Components
3. Contextual Information
4. Related Documentation
5. Technical Considerations

## Error Handling

### Common Scenarios
1. Insufficient Context
2. Model Timeouts
3. Invalid Input Format
4. Rate Limiting
5. Token Limits

### Response Types
1. No Answer Available
2. Conflict Resolution
3. Ambiguity Clarification
4. Error Explanation

## Related Documentation

- [Technical Reference](../docs/technical-reference.md) - System architecture
- [Utils Documentation](../docs/utils.md) - Utility functions
- [Frontend Documentation](../docs/frontend.md) - Interface implementation
- [Installation Guide](../docs/installation.md) - Setup instructions