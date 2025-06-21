# Agent Inbox Demo

## Demo Overview

This project demonstrates the powerful capabilities of LangChain's Agent Inbox by implementing a sophisticated web search and research agent. The system showcases Agent Inbox's ability to handle complex multi-step workflows with human-in-the-loop interactions. The agent performs web searches using Tavily, filters results for relevance, and generates comprehensive summaries - all while providing human oversight at critical decision points through Agent Inbox's interrupt system. This demo highlights how Agent Inbox can orchestrate complex AI workflows that combine automated processing with strategic human intervention, making it ideal for research tasks that require both AI efficiency and human judgment.

## Features

- **Multi-step Research Workflow**: Creates research plans, generates search queries, performs web searches, and summarizes results
- **Human-in-the-Loop**: Interactive checkpoints at each critical decision point
- **Intelligent Filtering**: AI-powered relevance filtering of search results
- **Web Search Integration**: Uses Tavily for comprehensive web search capabilities
- **Structured Output**: Generates well-organized research summaries

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory with your API keys:
```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=your_langsmith_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Running the Demo

### 1. Start LangGraph Development Server
```bash
# In the root directory of this project
langgraph dev
```

### 2. Start Agent Inbox Locally
```bash
# In a separate terminal, start Agent Inbox
cd agent-inbox
yarn build
yarn start
```

### 3. Configure Agent Inbox
1. Open Agent Inbox in your browser
3. Set the **Deployment URL** to: `http://127.0.0.1:2024`
4. Set the **Graph ID** to: `ambient`
5. Save the configuration

### 4. Start Using the Agent
1. Navigate to LangGraph Studio to enter a question, or make an API call to `http://127.0.0.1:2024`. API docs available at `http://127.0.0.1:2024/docs`
2. Enter your research question
3. From here, monitor your agent inbox for Human In the Loop Prompts to appear
4. Review and approve each step of the research workflow

The agent will perform web searches, filter results for relevance, and generate summaries while allowing you to intervene and provide feedback at each critical decision point through Agent Inbox's human-in-the-loop capabilities.

## Project Structure

- `ambient.py` - Main graph definition and workflow logic
- `tools.py` - Custom tools for web search and filtering
- `prompts.py` - Prompt templates for different workflow stages
- `utils.py` - Utility functions for state management