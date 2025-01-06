import logging
from pathlib import Path
from typing import List, Dict, Any

import yaml

from config import CONFIG


class PromptLoader:
    """Singleton class for loading prompts from YAML file."""
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

    def get_template(self, key: str) -> Any:
        """Get a template by key. Supports nested paths using dots."""
        try:
            value = self._prompts.get('templates', {})  # Start from templates
            # For red_calculator, look at root level
            if key.startswith('red_calculator.'):
                value = self._prompts

            for part in key.split('.'):
                value = value.get(part, {})

            # Special cases handling
            if key.endswith('.functions'):
                return value if isinstance(value, list) else []

            # For regular templates
            if isinstance(value, (dict, list)):
                return ''
            return value if value else ''

        except Exception as e:
            logging.error(f"Error getting template {key}: {e}")
            return ''

    def format_template(self, template_key: str, **kwargs) -> str:
        """Format a template with provided kwargs."""
        template = self.get_template(template_key)
        if not template:  # Handle empty string case
            logging.error(f"Template '{template_key}' not found or empty")
            return ''

        if not isinstance(template, str):
            logging.error(f"Template '{template_key}' is not a string: {template}")
            return ''

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logging.error(f"Missing required template parameter: {e}")
            return ''  # Return empty string instead of raising
        except Exception as e:
            logging.error(f"Error formatting template: {e}")
            return ''  # Return empty string instead of raising

    def get_no_answer_prompt(self) -> str:
        """Get the no-answer prompt."""
        return self.get_template('no_answer_prompt')

    def get_conflict_resolution_prompt(self) -> str:
        """Get the conflict resolution prompt."""
        return self.get_template('conflict_handling_prompt')

    def get_ambiguity_handling_prompt(self) -> str:
        """Get the ambiguity handling prompt."""
        return self.get_template('ambiguity_handling_prompt')


class PromptBuilder:
    """Class for building various types of prompts using the PromptLoader."""

    def __init__(self):
        self.loader = PromptLoader()

    def build_chat_prompt(
            self,
            query_text: str,
            contexts: List[str],
            images: List[Dict],
            chat_history: List[Dict],
            is_general: bool = True,
            is_technical: bool = False,
            is_summary: bool = False,
            is_overview: bool = False,
    ) -> str:
        """Build a complete prompt for the chat interaction."""
        # Process context information
        context_text = "\n\n".join(contexts) if contexts else "No relevant technical documentation found."

        # Process chat history
        chat_context = ""
        if chat_history:
            last_exchanges = chat_history[-(2 * CONFIG.MAX_CHAT_HISTORY):]
            history_entries = []
            for msg in last_exchanges:
                history_entries.append(
                    self.loader.format_template(
                        'chat_history_entry',
                        role='User' if msg['role'] == 'user' else 'assistant',
                        content=msg['content']
                    )
                )
            chat_context = "\nRecent Chat History:\n" + "\n".join(history_entries)

        # Process image information
        image_context = ""
        if images:
            image_descriptions = []
            for img in images:
                desc = self.loader.format_template(
                    'image_description',
                    source=img['source'],
                    caption_text=f": {img['caption']}" if img.get('caption') else "",
                    context_text=f" (Context: {img['context']})" if img.get('context') else ""
                )
                image_descriptions.append(desc)
            image_context = "\n\nRelevant Images:\n" + "\n".join(image_descriptions)

        # Combine instructions
        instructions = self.loader.get_instructions('general')
        if is_technical:
            instructions.extend(self.loader.get_instructions('technical'))
        elif is_summary:
            instructions.extend(self.loader.get_instructions('summary'))
        elif is_overview:
            instructions.extend(self.loader.get_instructions('overview'))

        # Build final prompt using template
        return self.loader.format_template(
            'chat_prompt',
            query_text=query_text,
            context_text=context_text,
            image_context=image_context,
            chat_context=chat_context,
            instructions="\n".join(instructions)
        )

    def build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build the messages list for the API request."""
        return [
            {
                "role": "system",
                "content": self.loader.get_system_prompt('technical_assistant')
            },
            {"role": "user", "content": prompt}
        ]

    def build_no_answer_message(self, query_text: str) -> List[Dict[str, str]]:
        """Build a no-answer message if no relevant information is found."""
        no_answer_prompt = self.loader.get_no_answer_prompt()
        formatted_no_answer = no_answer_prompt.format(query=query_text)
        return [
            {
                "role": "system",
                "content": self.loader.get_system_prompt('technical_assistant')
            },
            {"role": "user", "content": formatted_no_answer}
        ]
