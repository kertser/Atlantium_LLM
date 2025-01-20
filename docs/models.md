# Models Documentation

## Overview

The models package manages AI model integrations, prompt engineering, and response generation. It combines CLIP and BLIP for multimodal embeddings with GPT models for context-aware responses.

## Core Components

### PromptLoader
A singleton class managing system prompts and templates.
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
A class for constructing various types of prompts.
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

### CLIP/BLIP Integration
Used for multimodal embeddings and image analysis.
```python
def CLIP_init(model_name: str = "jinaai/jina-clip-v2"):
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

## Configuration

### Model Settings
- **CLIP_MODEL_NAME**: "jinaai/jina-clip-v2"
- **BLIP_MODEL_NAME**: "Salesforce/blip-image-captioning-base"
- **BASE_LLM_MODEL**: "gpt-4o-mini"
- **VISION_MAX_TOKENS**: 4096
- **EMBEDDING_DIMENSION**: 1024
- **USE_GPU**: True

### Thresholds
- **MINIMUM_TEXT_SIMILARITY**: 0.6
- **MINIMUM_IMAGE_SIMILARITY**: 0.25
- **IMAGE_SIMILARITY_THRESHOLD**: 0.3
- **TECHNICAL_CONFIDENCE_THRESHOLD**: 0.6
- **DEDUPLICATION_THRESHOLD**: 0.70

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

## Document Processing Settings
- **CHUNK_OVERLAP**: 100 words
- **MIN_CHUNK_SIZE**: 50 words
- **CHUNK_SIZE**: 400 words
- **MAX_TEXT_LENGTH**: 10,000
- **MAX_METADATA_SIZE**: 1,000,000 bytes
- **METADATA_TEXT_LIMIT**: 1500
- **COMPRESSION_ENABLED**: True
- **CLEANUP_FREQUENCY**: 10

## Related Documentation

- [Technical Reference](../docs/technical-reference.md)
- [Utils Documentation](../docs/utils.md)
- [Frontend Documentation](../docs/frontend.md)
- [Installation Guide](../docs/installation.md)

