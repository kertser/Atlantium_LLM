from duckduckgo_search import DDGS
from typing import Dict, Optional, List


class WebSearchAgent:
    """
    Web search agent for generating responses and summaries.
    Uses DuckDuckGo's AI chat for generating relevant responses.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the web search agent

        Args:
            model: Default AI model to use for chat responses
        """
        self.ddgs = DDGS()
        self.default_model = model

    def get_response(self,
                     query: str,
                     context: Optional[str] = None,
                     model: Optional[str] = None,
                     max_results: int = 10) -> Dict[str, str]:
        """
        Get AI-powered response for a query.
        If context is provided, retrieve the context from the web search and use AI chat to summarize it.
        If no context is provided, get results from the web search.

        Args:
            query: User's question
            context: Optional context to enhance the query
            model: Override default model choice
            max_results: Maximum number of search results to retrieve

        Returns:
            Dictionary containing response and metadata
        """
        try:
            if context:
                # Get search results from DuckDuckGo
                results = self.ddgs.text(
                    keywords=context,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit='y',
                    max_results=max_results
                )

                # Generate a summary from the search results
                summary = self.summarize_results(results)
                enhanced_query = f"In the context of {summary}, {query}"
            else:
                enhanced_query = query

            # Get response from DuckDuckGo chat
            response = self.ddgs.chat(
                keywords=enhanced_query,
                model=model or self.default_model,
                timeout=30
            )

            return {
                'status': 'success',
                'response': response,
                'source': 'web_search',
                'model_used': model or self.default_model
            }

        except Exception as e:
            return {
                'status': 'error',
                'response': f"Failed to get web search response: {str(e)}",
                'source': 'web_search',
                'model_used': model or self.default_model
            }

    @staticmethod
    def summarize_results(results: List[Dict[str, str]]) -> str:
        """
        Generate a summary from search results

        Args:
            results: List of search result dictionaries

        Returns:
            Summary string
        """
        if not results:
            return "No results found."

        summary = []
        for result in results:
            title = result.get("title", "No title")
            body = result.get("body", "No description available")
            summary.append(f"{title}: {body}")

        return "\n".join(summary)


if __name__ == "__main__":
    # Test the agent directly
    agent = WebSearchAgent()

    # Test web search summary (no context provided)
    result = agent.get_response(
        query="How cold is it?"  # No context is provided, therefore AI only
    )
    print(result['response'])

    print("-" * 80 + '\n')

    # Test chat response with context
    result = agent.get_response(
        query="Weather in israel",  # AI-bases summary by query
        context="Weather in israel"  # What shall be searched in the web
    )
    print(result['response'])
