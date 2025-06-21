from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from typing import List

# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=3)

@tool
def filter_by_relevance(results: List[Document], relevancy: List[bool]) -> List[Document]:
    """
    Filter results based on relevancy.
    Args:
        results: List of documents to filter
        relevancy: List of booleans indicating if the result is relevant
    Returns:
        List of documents that are relevant
    """
    return [result for result, relevant in zip(results, relevancy) if relevant]

@tool
def return_results() -> str:
    """
    Finalize the results and indicate that results are ready to be returned to the user.
    """
    return "Complete"