from duckduckgo_search import DDGS
from typing import Dict, Optional


class WebSearchAgent:
    """
    Fallback web search agent for when RAG returns no results.
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
                     model: Optional[str] = None) -> Dict[str, str]:
        """
        Get AI-powered response for a query

        Args:
            query: User's question
            context: Optional context about Atlantium Technologies
            model: Override default model choice

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Add context if provided
            if context:
                enhanced_query = f"In the context of {context}, {query}"
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


# Usage example in your RAG system:
"""
from models.agents.websearch_agent import WebSearchAgent

class RAGSystem:
    def __init__(self):
        self.web_search = WebSearchAgent(model="claude-3-haiku")  # Initialize with preferred model
        # ... other RAG system initialization ...

    async def get_answer(self, query: str) -> Dict:
        # First try RAG
        rag_results = self.retrieve_documents(query)

        if not rag_results:  # If RAG returns no documents
            # Use web search as fallback
            context = "Atlantium Technologies, a company specializing in UV water treatment solutions"
            web_result = self.web_search.get_response(
                query=query,
                context=context
            )
            return {
                'answer': web_result['response'],
                'source': 'web_search',
                'model': web_result['model_used']
            }

        # Continue with normal RAG processing if documents were found
        return self.process_rag_results(rag_results)
"""

if __name__ == "__main__":
    # Test the agent directly
    agent = WebSearchAgent()
    result = agent.get_response(
        query="What are Atlantium's main water treatment technologies?",
        context="Atlantium Technologies is a water treatment company"
    )
    print(result['response'])
