from langchain_core.tools import tool, InjectedToolCallId
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from typing import List, Annotated
from langgraph.types import Command
from langchain_core.messages import ToolMessage

# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=3)

@tool
def filter_by_relevance(results: list[Document], relevancy: List[bool]) -> List[Document]:
    """
    Filter results based on relevancy. Requires conversion of Tavily search results to Document objects.
    Args:
        results: List of results to filter. Must be a list of Document objects.
        relevancy: List of booleans indicating if the result is relevant
    Returns:
        List of documents that are relevant
    """
    documents = [doc for doc, relevant in zip(results, relevancy) if relevant]
    return documents