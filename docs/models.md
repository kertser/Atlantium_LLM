# Models Documentation

## Overview

The models package handles AI model management, prompt engineering, and response generation. It integrates CLIP for multimodal embeddings and interfaces with various LLM providers.

## Components

### PromptLoader (`prompt_loader.py`)

Manages prompt templates and system instructions:

```python
class PromptLoader:
    def get_system_prompt(key: str) -> str
    def get_instructions(instruction_type: str) -> List[str]
    def get_template(key: str) -> str
```

### PromptBuilder (`prompts.py`)

Constructs prompts for different query types:

```python
class PromptBuilder:
    def build_chat_prompt(
        query_text: str,
        contexts: List[str],
        images: List[Dict],
        chat_history: List[Dict],
        is_technical: bool = False
    ) -> str
```

### Prompt Templates (`templates/prompts.yaml`)

Defines system behavior and response formatting:
- Technical assistant configuration
- Response templates
- Formatting rules
- Error handling

## Usage Examples

### Building a Chat Prompt
```python
from models.prompts import PromptBuilder

builder = PromptBuilder()
prompt = builder.build_chat_prompt(
    query_text="How does UV disinfection work?",
    contexts=relevant_docs,
    images=relevant_images,
    chat_history=[]
)
```

### Loading System Prompts
```python
from models.prompt_loader import PromptLoader

loader = PromptLoader()
system_prompt = loader.get_system_prompt('technical_assistant')
```

## Configuration

Key settings in `prompts.yaml`:
- Model behavior configuration
- Response formatting rules
- System instructions
- Template definitions

## Custom Prompt Development

1. Edit `prompts.yaml` to add new templates:
```yaml
templates:
  your_template:
    content: |
      Your template content here
      {variable_placeholder}
```

2. Use in code:
```python
loader = PromptLoader()
formatted = loader.format_template('your_template', 
    variable_placeholder='value')
```

## Future Development

Planned extensions:
- Fine-tuned UV technical domain models
- Additional response templates
- UV Systems Calculator Agent (microservice API)
- Improved context management
