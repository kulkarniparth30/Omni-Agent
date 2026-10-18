# Omni-Agent 🤖

> Production-grade, asynchronous AI Agent & RAG platform powered by **LangGraph**, **FastAPI**, **FAISS**, and **Server-Sent Events (SSE)**.

---

## 🌟 Overview

**Omni-Agent** is an enterprise-ready, modular AI agent service refactored from a monolithic script into a clean, tiered architecture. It brings together modern LLMs with dynamic tool execution, disk-persisted Retrieval-Augmented Generation (RAG), and a headless REST/Streaming API consumed by a lightweight Streamlit client.

### Key Capabilities
- 🔄 **LangGraph State Machine**: Declarative conversation graph with conditional edges, automatic tool routing, and async SQLite state checkpointing.
- ⚡ **FastAPI Asynchronous Gateway**: Native ASGI server using Uvicorn with full Lifespan management for connection pooling and graceful teardown.
- 📡 **Real-Time Token Streaming (SSE)**: Streams tokens and tool execution lifecycles (`tool_start`, `tool_end`, `token`, `done`) via Server-Sent Events.
- 🛠️ **Multi-Tool Capability**:
  - **Alpha Vantage**: Real-time stock market quotes.
  - **DuckDuckGo**: Live web search.
  - **Semantic Scholar**: Academic research papers and literature search.
  - **Model Context Protocol (MCP)**: Arithmetic via stdio, plus remote tools via streamable HTTP.
  - **Per-Thread Document RAG**: Automatic parameter injection via LangChain's `RunnableConfig` (no LLM hallucination of thread IDs).
- 💾 **Persistent Vector Indexing**: FAISS vector indices and metadata are serialized to disk (`data/indexes/{thread_id}/`) and lazy-loaded on demand.
- 📊 **3-Tier Evaluation Suite**: Automated benchmarks modeled after the **CampusX / DeepEval** architecture spanning Component Evals, the RAG Triad, G-Eval Application Quality, Safety Gates, and Operational Latency.

---

## 🏗️ Architecture

```
Omni-Agent/
├── app/
│   ├── main.py              # FastAPI app + lifespan (startup/shutdown)
│   ├── config.py            # Centralized settings via pydantic-settings
│   ├── api/
│   │   ├── deps.py          # Dependency injection: get_agent(), get_checkpointer()
│   │   └── routes/
│   │       ├── chat.py      # POST /chat, POST /chat/stream (SSE)
│   │       ├── threads.py   # GET /threads, GET /threads/{id}/messages
│   │       ├── documents.py # POST /threads/{id}/documents (PDF upload)
│   │       └── health.py    # GET /health
│   ├── schemas/             # Pydantic request and response models
│   ├── agent/
│   │   ├── graph.py         # Pure build_graph() factory
│   │   ├── state.py         # ChatState with Sequence[BaseMessage]
│   │   └── llm.py           # ChatGroq & GoogleGenerativeAIEmbeddings factories
│   ├── tools/
│   │   ├── __init__.py      # Tool registry: get_all_tools()
│   │   ├── search.py        # DuckDuckGo & Semantic Scholar
│   │   ├── stock.py         # Alpha Vantage stock quotes
│   │   ├── rag.py           # Per-thread RAG via RunnableConfig
│   │   └── mcp.py           # Model Context Protocol client & async loader
│   ├── services/
│   │   └── pdf_store.py     # PDF chunking, FAISS disk persistence & lazy-loader
│   ├── prompts/
│   │   └── system.py        # Dynamic system prompt builder
│   └── core/                # Request-ID correlation middleware, exceptions, logging
├── ui/
│   └── streamlit_app.py     # Thin client communicating purely over HTTP & SSE
├── evals/                   # CampusX / DeepEval 3-tier evaluation framework
│   ├── metric_registry.py   # Thresholds, guardrails, and release gates
│   ├── harness.py           # Golden dataset loader & fixture uploader
│   ├── eval_retriever.py    # Component: Contextual Recall & Precision
│   ├── eval_generator.py    # Component: Generator Faithfulness & Relevancy
│   ├── eval_rag_pipeline.py # Pipeline: The full RAG Triad
│   ├── eval_application.py  # Application: Tool selection accuracy & G-Eval
│   ├── eval_safety.py       # Safety: Scope jailbreak, Prompt leakage, Toxicity
│   ├── eval_ops.py          # Operations: p50/p95 Latency & Reliability
│   ├── run_suite.py         # Master orchestrator writing snapshot JSONs
│   ├── compare.py           # Regression comparator (baseline vs candidate)
│   └── goldens/             # 30+ golden test cases and test PDF fixture
├── tests/                   # Automated pytest integration test suite
├── .env.example             # Configuration template
└── requirements.txt         # Pinned production dependencies
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment
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
Create your `.env` file by copying the template:
```bash
cp .env.example .env
```
Fill in your active API keys inside `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GOOGLE_API_KEY=your_google_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### 3. Launch the Backend API
```powershell
uvicorn app.main:app --reload --port 8000
```
Open your browser at **`http://localhost:8000/docs`** to view the interactive Swagger API documentation.

### 4. Launch the Streamlit Frontend
In a separate terminal window:
```powershell
streamlit run ui/streamlit_app.py --server.port 8501
```
Open **`http://localhost:8501`** to interact with Omni-Agent.

---

## 📡 API Endpoint Recipes (curl Examples)

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### 2. Synchronous Chat Turn (Non-Streaming)
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the stock price of Apple?", "thread_id": "thread-001"}'
```

### 3. Real-Time Streaming Chat Turn (Server-Sent Events)
```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{"message": "Search the web for news about Artemis mission.", "thread_id": "thread-001"}'
```

### 4. Upload and Index a PDF for a Thread
```bash
curl -X POST "http://localhost:8000/api/v1/threads/thread-001/documents" \
     -F "file=@sample.pdf"
```

### 5. List All Active Conversation Threads
```bash
curl -X GET "http://localhost:8000/api/v1/threads"
```

### 6. Retrieve Full Message History for a Thread
```bash
curl -X GET "http://localhost:8000/api/v1/threads/thread-001/messages"
```

---

## 📊 Automated Evaluation Suite (CampusX / DeepEval)

Omni-Agent includes a production benchmark suite measuring **Component**, **Pipeline (RAG Triad)**, **Application**, and **Safety** metrics.

### Run Full Suite and Generate a Baseline:
```powershell
python -m evals.run_suite --baseline
```

### Run a Regression Comparison:
```powershell
python -m evals.compare --base evals/baselines/baseline.json --cand evals/baselines/candidate.json
```

---

## 🧪 Running Automated Unit Tests
```powershell
pytest tests/ -v
```

---

## 📄 License
Licensed under the Apache 2.0 License.
