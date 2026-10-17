# ToolBot Refactor: Antigravity Prompt

> Paste this whole file into Antigravity. If it rushes, paste one step at a time.

---

## Role

You are a senior AI backend engineer and **teacher**. I am a 3rd-year CS student. My goal is not only to finish this refactor but to **learn how it is done**, so I can repeat it on my own projects and explain it in interviews.

For every step: explain in **plain language first**, then show the technical detail and the code.

---

## Teaching Mode (applies to EVERY step)

Before and after each step, use this exact format:

### Before coding
1. **What we are doing**: one or two plain-language sentences.
2. **Why it matters**: what problem this solves in my current code.
3. **Concepts I need**: explain any new concept (for example lifespan, dependency injection, SSE, Pydantic schemas) from scratch with a small, simple example before using it.

### While coding
4. **Show the code in small pieces**, not one giant dump. After each piece, explain line by line what it does and **why it was written that way**.
5. **Before / After comparison**: show how my original code looked and where each piece moved to. For example: "`get_stock_price` was in `Main_Backend.py` lines X to Y, now it is in `app/tools/stock.py`".
6. **Alternatives**: mention at least one other way to do it and why you did not choose it.

### After coding
7. **How to verify**: exact commands to run, and what output I should see.
8. **Common mistakes**: 2 or 3 mistakes beginners make in this step and how to avoid them.
9. **Try it yourself**: one small exercise I can do on my own (for example "add a new tool called X and register it").
10. **What I learned**: 3 or 4 lines I can say in an interview.
11. **Check my understanding**: ask me 2 short questions about this step. Do **not** move to the next step until I answer or say "continue".

---

## Context

I have a project called **ToolBot** with two files: `Main_Backend.py` and `Main_Frontend.py` (in the workspace).

- **Stack**: LangGraph (`StateGraph` + `ToolNode` + `AsyncSqliteSaver`), `ChatGroq` (llama-3.3-70b-versatile), Gemini embeddings, FAISS, Streamlit UI.
- **Tools**: DuckDuckGo search, Semantic Scholar, Alpha Vantage stock price, a per-thread PDF RAG tool, and MCP tools (arith via stdio, plus expense, context7 and deepwiki via streamable_http).

## Goal

Refactor into a clean project with a **FastAPI API layer**, separate `tools/` and `prompts/` modules, and an `evals/` folder, **without breaking current behavior**. Streamlit must keep working, but it should talk to the API instead of importing backend code.

---

## Step 0: Analysis only (no code changes)

- Read both files and explain the current architecture and the request flow in plain language. Include a simple diagram (text or mermaid) of how a user message travels today.
- List the problems in the current code. I already know about these, so explain each one and why it is a problem:
  - A hardcoded API key in `get_stock_price` and a hardcoded MCP file path.
  - Everything runs at import time (checkpointer, MCP loading, event-loop thread).
  - `_THREAD_RETRIEVERS` and `_THREAD_METADATA` are in-memory only, so PDFs are lost on restart while chat history survives in SQLite.
  - `rag_tool` relies on the LLM to pass `thread_id` correctly.
  - `ChatState` types messages as `list[BaseModel]` instead of `BaseMessage`.
  - `chat_node` swallows exceptions into a fake AIMessage.
  - The Streamlit frontend imports backend internals directly.
- Propose the folder structure below and explain what each folder and file is for.
- **Stop and wait for my approval.**

## Step 1: Target folder structure

```
toolbot/
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
│   │   ├── graph.py         # build_graph(tools, checkpointer) -> compiled graph
│   │   ├── state.py         # ChatState
│   │   └── llm.py           # ChatGroq + embeddings factory
│   ├── tools/
│   │   ├── __init__.py      # get_all_tools() registry
│   │   ├── search.py        # DuckDuckGo + Semantic Scholar
│   │   ├── stock.py         # get_stock_price (key from config)
│   │   ├── rag.py           # rag_tool
│   │   └── mcp.py           # MCP client + loader (servers from config)
│   ├── services/
│   │   └── pdf_store.py     # ingest_pdf, retriever registry
│   ├── prompts/
│   │   └── system.py        # system prompt builder
│   └── core/                # logging, exceptions, middleware
├── ui/
│   └── streamlit_app.py     # calls the API with httpx, no backend imports
├── evals/
│   ├── datasets/            # tool_selection.jsonl, rag_qa.jsonl, edge_cases.jsonl
│   ├── metrics.py
│   ├── run_evals.py
│   └── results/             # timestamped JSON reports
├── tests/                   # pytest API tests
├── .env.example
├── requirements.txt
└── README.md
```

Explain the **rule behind this layout**: routes handle HTTP only, services and agent hold logic, tools are isolated, prompts are separate, nothing imports upward. Show me which folder may import which.

## Step 2: Refactor the agent core (no FastAPI yet)

- Move each tool to its own file and register them in `tools/__init__.py`.
- Move the system prompt to `prompts/system.py` as `build_system_prompt(has_document: bool)`.
- `agent/graph.py` must contain **no FastAPI code** and **no import-time side effects**. It exposes `build_graph(tools, checkpointer)`.
- Remove the hardcoded key and path. Read both from `config.py`.
- Fix `ChatState` to use `BaseMessage`.
- For `rag_tool`, stop trusting the LLM for `thread_id`. Read it from `RunnableConfig` inside the tool, so the model only supplies `query`.
- After each file, explain **why** it is separated this way and what would break if it were not.

## Step 3: FastAPI layer

- Use FastAPI's **lifespan** to create the async SQLite connection, checkpointer, MCP tools and compiled graph **once at startup**, store them on `app.state`, and close them on shutdown. This **replaces my background-thread event loop** (`_ASYNC_LOOP`). Explain in detail why this is better and how the event loop works.
- Endpoints:
  - `POST /chat`: `{message, thread_id}` returns `{response, tools_used, thread_id, latency_ms}`.
  - `POST /chat/stream`: SSE streaming of tokens plus tool events (`tool_start`, `tool_end`, `token`, `done`).
  - `POST /threads/{id}/documents`: PDF upload and indexing, returns filename, pages and chunks.
  - `GET /threads` and `GET /threads/{id}/messages`: replaces `retrieve_all_threads` and `load_conversation`.
  - `GET /health`.
- Pydantic validation, proper status codes (404 unknown thread, 422 bad body, 502 LLM failure), a global exception handler, CORS, and request logging with a request id.
- Stop swallowing LLM errors into a fake AIMessage. Raise them and let the API return a proper error.
- Use `thread_id` as the one identifier shared by chat, PDF and checkpointer.
- Trace one full request: HTTP, route, schema, graph, tool, response. Show it as a diagram.
- Show me how to test every endpoint using the auto-generated `/docs` page **and** curl.

## Step 4: Persist the PDF index

- Save FAISS indexes per thread to disk (`data/indexes/{thread_id}`) and reload them on startup, so PDFs survive restarts. Keep the interface in `services/pdf_store.py` so storage can be swapped later (for example to ChromaDB).

## Step 5: Streamlit becomes a thin client

- `ui/streamlit_app.py` uses `httpx` to call the API (including SSE for streaming). It must not import anything from `app/`.
- Keep the existing features: new chat, past threads, PDF upload, tool-status indicator.
- Explain why a thin client is better than importing the backend.

## Step 6: Evals

- `datasets/`: at least 30 JSONL cases with `id`, `category`, `input`, `expected_tools`, `expected_answer_contains`, and an optional `pdf_fixture`. Categories:
  - **Tool selection**: stock price, web search, paper search, math via MCP, PDF question.
  - **No tool needed**: greetings, simple facts.
  - **RAG**: questions answerable from a small fixture PDF, plus a question that is **not** in the PDF.
  - **Failure/edge**: invalid stock symbol, a RAG question with no PDF uploaded, an ambiguous query.
- `metrics.py`:
  - Tool selection accuracy.
  - Tool-call correctness (right arguments).
  - Answer correctness (keyword match, plus an optional LLM-as-judge using a different model than the agent).
  - Hallucination check on RAG (does the answer stay grounded in the retrieved context?).
  - Latency (p50 and p95) and error rate.
- `run_evals.py`: calls the **running API** (or the graph directly with `--direct`), prints a summary table, and saves a timestamped JSON report to `results/`. Add a `--category` filter and a `--compare` flag to diff two result files.
- Mock external tools where needed so evals are repeatable.
- **Teach me evals properly**: what an eval is, why LLM apps need them, how to design good test cases, what each metric means, and how to read the results and decide what to improve.

## Step 7: Tests and docs

- A few pytest tests for the API (health, chat success with a mocked graph, 404 thread, 422 validation).
- README: setup, `.env.example`, `uvicorn app.main:app --reload`, curl examples for every endpoint, how to run the UI, how to run evals.

## Final Step: Recap

- A one-page summary of the whole journey: before vs after architecture, every concept I learned, and a list of 10 likely interview questions about this project with short model answers.
- Suggest 3 improvements I can build on my own next (for example auth, rate limiting, Docker).

---

## Rules

- Work in small steps. **Stop after each step** and wait for my review.
- Don't change behavior beyond what is listed. Don't add libraries beyond `fastapi`, `uvicorn`, `pydantic-settings`, `httpx`, `sse-starlette` (if needed) and `pytest`.
- Never skip the Teaching Mode sections. If a step feels too small to explain, explain it anyway.
- Use simple words. Define every technical term the first time you use it.
- If something I asked for is a bad idea, say so and suggest an alternative.
- Give me exercises so I write some code myself instead of only reading yours.
