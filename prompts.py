from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List


def get_research_plan_prompt(question: str):
    system = """
    You are a helpful assistant that takes a question and generates a research plan to answer the question.
    Your research plan should include be 3 steps, and offer a thorough approach to investigate
    and answer the question. The research plan should include multiple appraoches in case one doesn't work.

    Return the research plan in the following format:

    Step 1: <step 1 description>
    Step 2: <step 2 description>
    Step 3: <step 3 description>

    Do not include any other text in your response.
    """
    user = """
    Can you help me research the following question: {question}
    """
    user = user.format(question=question)
    prompts = [SystemMessage(content=system), HumanMessage(content=user)]
    return prompts


def get_search_query_prompts(research_plan: str):
    system = """
    You are a helpful assistant that takes research plan, and generates
    optimal search queries to answer the question.

    Create a 3 search queries to help achieve the given research plan

    Optimize these queries to return good results from Google.

    Return the search queries as a list of strings.
    These search queries should be specific and relevant to the question.
    These search queries MUST be diverse and cover different approaches to the question.
    """
    user = """
    Research Plan: {plan}
    """
    user = user.format(plan=research_plan)

    prompts = [SystemMessage(content=system), HumanMessage(content=user)]
    return prompts

def get_search_assistant_prompt(search_queries: List[str]):
    system = """
    You are an expert researcher. You have been given a list of search queries to execute. 

    Your job is to execute the search queries, and return the relevant results. 
    You should use your best judgement to determine if the result is credible and relevant.

    You have the following tools at your disposal:
    - Web Search: This tool allows you to search the web for information.
    - Filter by Relevance: This tool allows you to filter the results based on relevancy.
    - Return Results: This tool allows you to return your results to the user.
    """
    user = """
    Search Queries: 
    {queries}
    """
    user = user.format(queries="\n".join(search_queries))
    prompts = [SystemMessage(content=system), HumanMessage(content=user)]
    return prompts

def summarize_assistant_prompt(question: str, research_plan: str, results: List[str]):
    system = """
    You are an expert researcher. You have been given a list of results collected from a thorough research plan.
    The research plan was designed to answer a specific question. 
    Summarize the results in a way that is helpful to answer the question.

    Question: {question}

    Research Plan: {research_plan}

    Results: {results}
    """
    user = """
    Please summarize the results to answer the question in structured, systematic way.
    """
    system = system.format(question=question, research_plan=research_plan, results="\n".join(results))
    prompts = [SystemMessage(content=system), HumanMessage(content=user)]
    return prompts