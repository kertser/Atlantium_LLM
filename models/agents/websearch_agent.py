from duckduckgo_search import DDGS
from typing import Dict, Optional, List


class WebSearchAgent:
    """
    Web search agent for generating responses and summaries.
    Uses DuckDuckGo's AI chat for generating relevant responses.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_results: int = 10):
        """
        Initialize the web search agent

        Args:
            model: Default AI model to use for chat responses
            max_results: Maximum number of search results to retrieve
        """
        self.ddgs = DDGS()
        self.default_model = model
        self.max_results = max_results

    def get_response(
            self,
            query: str,
            context: Optional[str] = None,
            model: Optional[str] = None,
            max_results: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Get AI-powered response for a query.
        If context is provided, retrieve the context from the web search and use AI chat to summarize it.
        If no context is provided, get results from the web search.

        Args:
            query: User's question
            context: Optional context to enhance the query
            model: Override default model choice
            max_results: Maximum number of search results to retrieve, defaults to self.max_results

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Use instance max_results if not specified
            max_results = max_results if max_results is not None else self.max_results

            if context:
                try:
                    # Get search results from DuckDuckGo
                    results = list(self.ddgs.text(
                        keywords=context,
                        region='wt-wt',
                        safesearch='moderate',
                        timelimit='y',
                        max_results=max_results
                    ))

                    # Generate a summary from the search results
                    summary = self.summarize_results(results)
                    enhanced_query = f"In the context of {summary}, {query}"
                except Exception as search_error:
                    # If web search fails, fall back to just the query
                    enhanced_query = query
            else:
                enhanced_query = query

            # Get response from DuckDuckGo chat
            response = self.ddgs.chat(
                keywords=enhanced_query,  # Changed to keywords
                model=model or self.default_model,
                timeout=30
            )

            return {
                'status': 'success',
                'response': response,
                'source': 'web_search',
                'model_used': model or self.default_model,
                'enhanced_query': enhanced_query
            }

        except Exception as e:
            return {
                'status': 'error',
                'response': f"Failed to get web search response: {str(e)}",
                'source': 'web_search',
                'model_used': model or self.default_model,
                'error': str(e)
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

        # Limit the number of results to summarize to prevent overly long summaries
        MAX_RESULTS_TO_SUMMARIZE = 5
        summary_parts = []

        for result in results[:MAX_RESULTS_TO_SUMMARIZE]:
            title = result.get("title", "").strip()
            body = result.get("body", "").strip()

            if title and body:
                summary_parts.append(f"{title}: {body}")
            elif title:
                summary_parts.append(title)
            elif body:
                summary_parts.append(body)

        return " | ".join(summary_parts) if summary_parts else "No relevant information found."


if __name__ == "__main__":
    # Test the agent directly
    agent = WebSearchAgent()

    # Test web search summary (no context provided)
    result = agent.get_response(
        query="How cold is it?"  # No context is provided, therefore AI only
    )
    print("Direct query result:")
    print(result['response'])
    print("\n" + "-" * 80 + "\n")

    # Test chat response with context
    result = agent.get_response(
        query="What is the current temperature?",  # AI-based summary by query
        context="Weather in Tel Aviv today"  # What shall be searched in the web
    )
    print("Query with context result:")
    print(result['response'])