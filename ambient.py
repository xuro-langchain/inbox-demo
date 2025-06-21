import random 
import base64
import asyncio
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt, ActionRequest
from langgraph.types import interrupt, Command


from prompts import (
    get_research_plan_prompt, 
    get_search_query_prompts, 
    get_search_assistant_prompt, 
    summarize_assistant_prompt
)
from tools import (
    web_search_tool,
    return_results,
    filter_by_relevance
)


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
tools = [web_search_tool, return_results, filter_by_relevance]

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    research_plan: str
    search_queries: list[str]
    documents: list[Document]
    summary: str
    messages: Annotated[list, add_messages]
    error: str
    

class InputState(TypedDict):
    question: str

# Research Plan Creation Node
async def create_research_plan(state: GraphState):
    question = state["question"]
    prompts = get_research_plan_prompt(question)
    llm_response = await llm.ainvoke(prompts)
    return {"question": question, "research_plan": llm_response.content}
# ------------------------------------------------------------

# Search Query Creation Node
class SearchQueries(BaseModel):
    """Schema for parsing user-provided account information."""
    queries: List[str] = Field(description = "List of search queries optimized for Google")

async def create_search_queries(state: GraphState):
    research_plan = state["research_plan"]
    prompts = get_search_query_prompts(research_plan)
    structured_llm = llm.with_structured_output(SearchQueries)
    llm_response = await structured_llm.ainvoke(prompts)
    return {"search_queries": llm_response.queries}
# ------------------------------------------------------------

# Search Assistant with Tools
async def search_assistant(state: GraphState):
    search_queries = state["search_queries"]
    prompts = get_search_assistant_prompt(search_queries)
    llm_with_tools = llm.bind_tools(tools)
    llm_response = await llm_with_tools.ainvoke(prompts)
    return {"search_results": llm_response.content}

tool_node = ToolNode(tools)
# ------------------------------------------------------------

# Summarize Results Node
async def summarize_results(state: GraphState):
    question = state["question"]
    research_plan = state["research_plan"]
    results = state["search_results"]
    prompts = summarize_assistant_prompt(question, research_plan, results)
    llm_response = await llm.ainvoke(prompts)
    return {"summary": llm_response.content}
# ------------------------------------------------------------

config = HumanInterruptConfig(
    allow_accept=True,   # Allow human to approve/accept the current state
    allow_ignore=True,   # Allow skipping/ignoring this step
    allow_respond=False,  # Allow providing text feedback
    allow_edit=True     # Disallow editing the content/state
)

def generate_markdown(description: str, state: GraphState, diagram: str):
    markdown = f"# Instructions\n{description}\n\n"
    # markdown += f"## Graph Diagram\n![Graph Diagram]({diagram})\n\n"
    markdown += f"## State Snapshot\n"
    for key, value in state.items():
        markdown += f"### {key}\n{value}\n"
    return markdown

# Interrupt Nodes
async def confirm_research_plan(state: GraphState):
    snapshot = {
        "question": state["question"],
        "research_plan": state["research_plan"],
        "messages": state.get("messages", []),
    }
    markdown = generate_markdown(
        "Please review the research plan and approve or provide feedback.", 
        snapshot, 
        diagram if diagram else "No image available"
    )

    args = {"research_plan": state["research_plan"]}
    action_request = ActionRequest(action="Review Research Plan", args=args)
    human_interrupt = HumanInterrupt(
        action_request=action_request,
        config=config,
        description=markdown
    )
    response = interrupt(human_interrupt)[0]
    if response.action == "accept":
        pass
    elif response.action == "ignore":
        pass
    elif response.action == "respond":
        return {"research_plan": response["args"]["args"]["research_plan"]}

async def confirm_search_queries(state: GraphState):
    snapshot = {
        "question": state["question"],
        "research_plan": state.get("research_plan", ""),
        "search_queries": state.get("search_queries", []),
        "messages": state.get("messages", []),
    }
    markdown = generate_markdown(
        "Please review the search queries and approve or provide feedback.", 
        snapshot, 
        diagram if diagram else "No image available"
    )
    args = {"search_queries": state.get("search_queries", [])}
    action_request = ActionRequest(action="Review Search Queries", args=args)
    human_interrupt = HumanInterrupt(
        action_request=action_request,
        config=config,
        description=markdown
    )
    response = interrupt(human_interrupt)[0]
    if response.action == "accept":
        pass
    elif response.action == "ignore":
        pass
    elif response.action == "respond":
        return {"search_queries": response["args"]["args"]["search_queries"]}

async def confirm_search_results(state: GraphState):
    snapshot = {
        "question": state["question"],
        "research_plan": state.get("research_plan", ""),
        "search_queries": state.get("search_queries", []),
        "search_results": state.get("search_results", ""),
        "messages": state.get("messages", []),
    }
    markdown = generate_markdown(
        "Please review the search results and approve or provide feedback.", 
        snapshot, 
        diagram if diagram else "No image available"
    )
    args = {"search_results": state.get("search_results", [])}
    action_request = ActionRequest(action="Review Search Results", args=args)
    human_interrupt = HumanInterrupt(
        action_request=action_request,
        config=config,
        description=markdown
    )
    response = interrupt(human_interrupt)[0]
    if response.action == "accept": 
        pass
    elif response.action == "ignore":
        pass
    elif response.action == "respond":
        return {"search_results": response["args"]["args"]["search_results"]}

async def confirm_summary(state: GraphState):
    snapshot = {
        "question": state["question"],
        "research_plan": state.get("research_plan", ""),
        "search_queries": state.get("search_queries", []),
        "search_results": state.get("search_results", ""),
        "summary": state.get("summary", ""),
        "messages": state.get("messages", []),
    }
    markdown = generate_markdown(
        "Please review the summary and approve or provide feedback.", 
        snapshot, 
        diagram if diagram else "No image available"
    )
    args = {"summary": state.get("summary", "")}
    action_request = ActionRequest(action="Review Summary", args=args)
    human_interrupt = HumanInterrupt(
        action_request=action_request,
        config=config,
        description=markdown
    )
    response = interrupt(human_interrupt)[0]
    if response.action == "accept":
        pass
    elif response.action == "ignore":
        pass
    elif response.action == "respond":
        return {"summary": response["args"]["args"]["summary"]}

# ------------------------------------------------------------

# Graph Definition
graph = StateGraph(GraphState, input=InputState)

graph.add_node("create_research_plan", create_research_plan)
graph.add_node("create_search_queries", create_search_queries)
graph.add_node("search_assistant", search_assistant)
graph.add_node("summarize_results", summarize_results)

graph.add_node("confirm_research_plan", confirm_research_plan)
graph.add_node("confirm_search_queries", confirm_search_queries)
graph.add_node("confirm_search_results", confirm_search_results)
graph.add_node("confirm_summary", confirm_summary)

graph.add_edge(START, "create_research_plan")
graph.add_edge("create_research_plan", "confirm_research_plan")
graph.add_edge("confirm_research_plan", "create_search_queries")
graph.add_edge("create_search_queries", "confirm_search_queries")
graph.add_edge("confirm_search_queries", "search_assistant")
graph.add_edge("search_assistant", "confirm_search_results")
graph.add_edge("confirm_search_results", "summarize_results")
graph.add_edge("summarize_results", "confirm_summary")

memory = AsyncSqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)

image_bytes = app.get_graph().draw_mermaid_png()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
image_data_uri = f"data:image/png;base64,{image_base64}"
diagram = image_data_uri