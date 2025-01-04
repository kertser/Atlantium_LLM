from duckduckgo_search import DDGS
from typing import List, Dict, Optional


class WebSearch:
    """Web search and chat implementation using DuckDuckGo"""

    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Regular web search"""
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def chat_query(self,
                   query: str,
                   model: str = "gpt-4o-mini",
                   timeout: int = 30) -> Optional[str]:
        """
        Get an AI-powered response using DuckDuckGo chat

        Args:
            query: Question or topic to discuss
            model: AI model to use. Options:
                - "gpt-4o-mini" (default)
                - "claude-3-haiku"
                - "llama-3.1-70b"
                - "mixtral-8x7b"
            timeout: Request timeout in seconds (default: 30)

        Returns:
            AI response or None if error occurs
        """
        try:
            response = self.ddgs.chat(
                keywords=query,
                model=model,
                timeout=timeout
            )
            return response
        except Exception as e:
            print(f"Chat error: {str(e)}")
            return None


def main():
    searcher = WebSearch()

    print("Available models:")
    print("1. gpt-4o-mini (default)")
    print("2. claude-3-haiku")
    print("3. llama-3.1-70b")
    print("4. mixtral-8x7b")

    query = input("\nEnter your question: ")
    model = input("Choose model (press Enter for default): ").strip()

    # Map model choice to actual model name
    model_map = {
        "1": "gpt-4o-mini",
        "2": "claude-3-haiku",
        "3": "llama-3.1-70b",
        "4": "mixtral-8x7b"
    }

    chosen_model = model_map.get(model, "gpt-4o-mini")

    response = searcher.chat_query(query, model=chosen_model)

    if response:
        print("\nAI Response:")
        print(response)
    else:
        print("Failed to get response.")


if __name__ == "__main__":
    main()