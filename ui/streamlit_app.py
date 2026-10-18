import os
import json
import uuid
from typing import Generator, Dict, Any, List, Optional
import streamlit as st
import httpx

# ==========================================
# Configuration & API Client
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

class ApiClient:
    """Thin HTTP client communicating with the Omni-Agent FastAPI backend."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Check if the FastAPI backend service is reachable and healthy."""
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=3.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {"status": "offline", "tools_count": 0, "available_tools": []}

    def list_threads(self) -> List[str]:
        """Fetch all existing conversation thread IDs."""
        try:
            r = httpx.get(f"{self.base_url}/threads", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("threads", [])
        except Exception as exc:
            st.sidebar.error(f"Failed to fetch threads: {exc}")
        return []

    def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Fetch message history for a specific thread."""
        try:
            r = httpx.get(f"{self.base_url}/threads/{thread_id}/messages", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("messages", [])
        except Exception:
            pass
        return []

    def upload_pdf(self, thread_id: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Upload a PDF to be indexed for the given thread."""
        files = {"file": (filename, file_bytes, "application/pdf")}
        r = httpx.post(
            f"{self.base_url}/threads/{thread_id}/documents",
            files=files,
            timeout=60.0
        )
        if r.status_code == 200:
            return r.json()
        error_detail = r.json().get("error", r.text) if r.headers.get("content-type") == "application/json" else r.text
        raise RuntimeError(f"Upload failed ({r.status_code}): {error_detail}")

    def stream_chat(
        self,
        message: str,
        thread_id: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response tokens and tool lifecycle events from POST /chat/stream via SSE.
        Yields structured event dicts:
          - {"event": "tool_start", "tool_name": "..."}
          - {"event": "tool_end", "tool_name": "..."}
          - {"event": "token", "token": "..."}
          - {"event": "done", "thread_id": "..."}
        """
        payload = {"message": message, "thread_id": thread_id}
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/stream",
            json=payload,
            timeout=120.0
        ) as response:
            if response.status_code != 200:
                error_msg = response.read().decode("utf-8")
                yield {"event": "error", "message": f"Server returned error {response.status_code}: {error_msg}"}
                return

            current_event = "message"
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event = line.replace("event:", "", 1).strip()
                elif line.startswith("data:"):
                    data_str = line.replace("data:", "", 1).strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {"text": data_str}

                    if current_event == "tool_start":
                        yield {"event": "tool_start", "tool_name": data.get("tool_name", "tool")}
                    elif current_event == "tool_end":
                        yield {"event": "tool_end", "tool_name": data.get("tool_name", "tool")}
                    elif current_event == "token":
                        yield {"event": "token", "token": data.get("token", "")}
                    elif current_event == "done":
                        yield {"event": "done", "thread_id": data.get("thread_id")}
                    elif current_event == "error":
                        yield {"event": "error", "message": data.get("error", "Unknown streaming error")}

api = ApiClient(API_BASE_URL)

# ==========================================
# Streamlit Session State Initialization
# ==========================================
def generate_thread_id() -> str:
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

def reset_chat():
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    st.session_state["ingested_docs"].setdefault(new_id, {})

# ==========================================
# Sidebar UI
# ==========================================
st.set_page_config(page_title="Omni-Agent", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] h1 {
        text-align: center;
        font-size: 36px;
    }
    .thread-box {
        text-align: center;
        font-size: 14px;
        padding: 8px;
        background: #1e1e2e;
        border-radius: 8px;
        margin-bottom: 12px;
        word-break: break-all;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🤖 Omni-Agent")

# Backend Health Status Indicator
health = api.check_health()
if health.get("status") == "ok":
    st.sidebar.success(f"Backend Connected ({health.get('tools_count')} tools active)")
else:
    st.sidebar.error("⚠️ Backend Offline (Start: uvicorn app.main:app)")

st.sidebar.markdown(
    f"<div class='thread-box'><b>Current Thread:</b><br><code>{thread_key}</code></div>",
    unsafe_allow_html=True,
)

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Document Status Indicator
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 `{latest_doc.get('filename')}` indexed\n\n"
        f"({latest_doc.get('chunks')} chunks, {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF document indexed for this chat.")

# PDF File Uploader
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed.")
    else:
        with st.sidebar.status("Indexing PDF via API…", expanded=True) as status_box:
            try:
                summary = api.upload_pdf(
                    thread_id=thread_key,
                    file_bytes=uploaded_pdf.getvalue(),
                    filename=uploaded_pdf.name
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="✅ PDF indexed successfully!", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:
                status_box.update(label=f"❌ Error: {exc}", state="error", expanded=True)

# Past Conversations List
st.sidebar.subheader("Past Conversations")
past_threads = api.list_threads()
selected_thread: Optional[str] = None

if not past_threads:
    st.sidebar.caption("No saved conversations yet.")
else:
    for t_id in past_threads:
        label = f"💬 {t_id[:8]}…" if len(t_id) > 10 else f"💬 {t_id}"
        if st.sidebar.button(label, key=f"thread-btn-{t_id}", use_container_width=True):
            selected_thread = t_id

# ==========================================
# Main Chat UI
# ==========================================
st.title("Omni-Agent: Multi-Utility AI")
st.caption("Equipped with Web Search, Academic Papers, Stock Quotes, MCP, and Document RAG.")

# Render previous messages from history
for msg in st.session_state["message_history"]:
    role = msg.get("role", "assistant")
    if role in ["user", "assistant"]:
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

# Handle User Input
user_input = st.chat_input("Ask about your document, search the web, check stocks, or solve math...")

if user_input:
    # 1. Display user message immediately
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Assistant streaming container
    with st.chat_message("assistant"):
        status_container = {"box": None}
        accumulated_tokens: List[str] = []

        def token_generator() -> Generator[str, None, None]:
            for event in api.stream_chat(user_input, thread_key):
                etype = event.get("event")

                if etype == "tool_start":
                    tname = event.get("tool_name", "tool")
                    if status_container["box"] is None:
                        status_container["box"] = st.status(f"🔧 Using `{tname}` …", expanded=True)
                    else:
                        status_container["box"].update(label=f"🔧 Using `{tname}` …", state="running", expanded=True)

                elif etype == "tool_end":
                    tname = event.get("tool_name", "tool")
                    if status_container["box"] is not None:
                        status_container["box"].update(label=f"✅ Tool `{tname}` finished", state="running", expanded=False)

                elif etype == "token":
                    tok = event.get("token", "")
                    accumulated_tokens.append(tok)
                    yield tok

                elif etype == "error":
                    st.error(event.get("message", "An error occurred."))

            # Finalize status container once streaming ends
            if status_container["box"] is not None:
                status_container["box"].update(label="✅ All tools completed", state="complete", expanded=False)

        full_response = st.write_stream(token_generator())

    # Save full assistant response into history
    st.session_state["message_history"].append({"role": "assistant", "content": full_response})

# Handle switching to a past thread
if selected_thread and selected_thread != st.session_state["thread_id"]:
    st.session_state["thread_id"] = selected_thread
    server_messages = api.get_thread_messages(selected_thread)
    st.session_state["message_history"] = [
        {"role": m["role"], "content": m["content"]}
        for m in server_messages
        if m["role"] in ["user", "assistant"]
    ]
    st.session_state["ingested_docs"].setdefault(selected_thread, {})
    st.rerun()
