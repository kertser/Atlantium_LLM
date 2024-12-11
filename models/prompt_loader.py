from pathlib import Path
from typing import List
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

    def get_example(self, example_key: str) -> str:
        """Get an example by key."""
        return self._prompts.get('examples', {}).get(example_key, '')

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

    def get_no_answer_prompt(self) -> str:
        """Get the no-answer prompt."""
        return self.get_template('no_answer_prompt')

    def get_conflict_resolution_prompt(self) -> str:
        """Get the conflict resolution prompt."""
        return self.get_template('conflict_handling_prompt')

    def get_ambiguity_handling_prompt(self) -> str:
        """Get the ambiguity handling prompt."""
        return self.get_template('ambiguity_handling_prompt')
