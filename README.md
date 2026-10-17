# Omni-Agent 🤖

> Production-grade, asynchronous AI Agent & RAG platform powered by **LangGraph**, **FastAPI**, and **Server-Sent Events (SSE)**.

---

## 🌟 Overview

**Omni-Agent** (formerly ToolBot) is a modular AI agent service designed using a tiered, clean architecture. It bridges modern LLMs with real-world tool execution, persistent document retrieval (RAG), and a headless REST/Streaming API.

### Key Capabilities
- 🔄 **LangGraph Orchestration**: Declarative state graph with automatic tool routing, conditional edges, and async SQLite checkpointing.
- ⚡ **FastAPI & Uvicorn Backend**: Asynchronous REST endpoints with full Lifespan management for connection pooling and graceful teardown.
- 📡 **Real-time Streaming (SSE)**: Streams tokens and tool lifecycle events (`tool_start`, `tool_end`, `token`, `done`) via Server-Sent Events.
- 🛠️ **Diverse Tool Ecosystem**:
  - **DuckDuckGo**: Live web search.
  - **Semantic Scholar**: Academic research and literature search.
  - **Alpha Vantage**: Real-time stock market data.
  - **Model Context Protocol (MCP)**: Arithmetic via stdio, plus remote tools via streamable HTTP.
  - **Per-Thread Document RAG**: Automatic contextual parameter injection via LangChain's `RunnableConfig` (no LLM hallucination of thread IDs).
- 🛡️ **Defensive Engineering**: Request-ID correlation middleware, Pydantic input/output schemas, and domain-level error handling.

---

## 🏗️ Architecture

```
Omni-Agent/
├── app/
│   ├── main.py              # FastAPI app + lifespan (startup/shutdown)
│   ├── config.py            # pydantic-settings, reads .env
│   ├── api/
│   │   ├── deps.py          # get_agent(), get_checkpointer()
│   │   └── routes/
│   │       ├── chat.py      # POST /chat, POST /chat/stream (SSE)
│   │       ├── threads.py   # GET /threads, GET /threads/{id}/messages
│   │       ├── documents.py # POST /threads/{id}/documents (PDF upload)
│   │       └── health.py    # GET /health
│   ├── schemas/             # Pydantic request/response models
│   ├── agent/
│   │   ├── graph.py         # build_graph(tools, checkpointer)
│   │   ├── state.py         # ChatState with BaseMessage
│   │   └── llm.py           # ChatGroq + embeddings factory
│   ├── tools/
│   │   ├── __init__.py      # get_all_tools() registry
│   │   ├── search.py        # DuckDuckGo + Semantic Scholar
│   │   ├── stock.py         # Alpha Vantage stock quotes
│   │   ├── rag.py           # Secure per-thread RAG tool
│   │   └── mcp.py           # Model Context Protocol client
│   ├── services/
│   │   └── pdf_store.py     # Document ingestion & FAISS store registry
│   ├── prompts/
│   │   └── system.py        # Dynamic system prompt builder
│   └── core/                # Request-ID middleware, exceptions, logging
├── ui/                      # Streamlit frontend client (HTTP/SSE only)
├── evals/                   # Evaluation suite (datasets, metrics, runner)
├── tests/                   # Pytest test suite
├── .env.example             # Configuration template
└── requirements.txt         # Pinned dependencies
```

---

## 🚀 Quickstart

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/kulkarniparth30/Omni-Agent.git
cd Omni-Agent

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Fill in your API keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`, `ALPHA_VANTAGE_API_KEY`) in `.env`.

### 3. Run the API Server
```bash
uvicorn app.main:app --reload --port 8000
```
Visit the interactive Swagger UI at: **`http://localhost:8000/docs`**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health status and active tools list |
| `POST` | `/api/v1/chat` | Non-streaming chat turn with latency tracking |
| `POST` | `/api/v1/chat/stream` | Real-time Server-Sent Events (SSE) token stream |
| `GET` | `/api/v1/threads` | List all conversation sessions |
| `GET` | `/api/v1/threads/{id}/messages` | Retrieve conversation history for a thread |
| `POST` | `/api/v1/threads/{id}/documents` | Upload and index a PDF for a thread |

---

## 📄 License
This project is licensed under the Apache 2.0 License.
