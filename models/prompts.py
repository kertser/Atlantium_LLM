from typing import List, Dict
from config import CONFIG
from .prompt_loader import PromptLoader


class PromptBuilder:
    def __init__(self):
        self.loader = PromptLoader()

    def build_chat_prompt(
        self,
        query_text: str,
        contexts: List[str],
        images: List[Dict],
        chat_history: List[Dict],
        is_technical: bool = False
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
                        role='User' if msg['role'] == 'user' else 'Assistant',
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
        instructions = self.loader.get_instructions('base')
        if is_technical:
            instructions.extend(self.loader.get_instructions('technical'))

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

    def build_conflict_resolution_message(self, conflicting_docs: List[Dict]) -> List[Dict[str, str]]:
        """Build a message to handle conflicting document data."""
        conflict_prompt = self.loader.get_conflict_resolution_prompt()
        formatted_conflict = conflict_prompt.format(documents="\n".join(conflicting_docs))
        return [
            {
                "role": "system",
                "content": self.loader.get_system_prompt('technical_assistant')
            },
            {"role": "user", "content": formatted_conflict}
        ]

    def build_ambiguity_message(self, query_text: str) -> List[Dict[str, str]]:
        """Build a message to handle ambiguous queries."""
        ambiguity_prompt = self.loader.get_ambiguity_handling_prompt()
        formatted_ambiguity = ambiguity_prompt.format(query=query_text)
        return [
            {
                "role": "system",
                "content": self.loader.get_system_prompt('technical_assistant')
            },
            {"role": "user", "content": formatted_ambiguity}
        ]
