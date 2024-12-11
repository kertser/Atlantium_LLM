from pathlib import Path
from typing import Dict, Any, List
import logging
import yaml

class PromptLoader:
    _instance = None
    _prompts = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._prompts is None:
            self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        try:
            prompts_path = Path(__file__).parent / "templates" / "prompts.yaml"
            with open(prompts_path, 'r', encoding='utf-8') as file:
                self._prompts = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading prompts: {e}")
            raise

    def get_system_prompt(self, key: str) -> str:
        """Get a system prompt by key."""
        return self._prompts.get('system', {}).get(key, '')

    def get_instructions(self, instruction_type: str) -> List[str]:
        """Get instructions by type."""
        return self._prompts.get('instructions', {}).get(instruction_type, [])

    def get_template(self, key: str) -> str:
        """Get a template by key."""
        return self._prompts.get('templates', {}).get(key, '')

    def format_template(self, template_key: str, **kwargs) -> str:
        """Format a template with provided kwargs."""
        template = self.get_template(template_key)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logging.error(f"Missing required template parameter: {e}")
            raise
        except Exception as e:
            logging.error(f"Error formatting template: {e}")
            raise
