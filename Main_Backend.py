from __future__ import annotations
from langgraph.graph import StateGraph ,START ,END
from typing import TypedDict , Annotated , Literal , Any , Dict , Optional
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage , HumanMessage , BaseMessage ,AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_core.tools import tool ,BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
import aiosqlite
import requests
import operator
import asyncio
import threading
import os 
import tempfile

load_dotenv()

# ********************* Async Loop for backend tasks *******************

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever , daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro , _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-001",
#     temperature=0, 
#     streaming=True
# )

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    streaming=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ***************************** PDF retriever store {per thread} ********************************

_THREAD_RETRIEVERS : Dict[str , Any] ={}
_THREAD_METADATA : Dict[str , dict] = {}

def _get_retriever(thread_id : Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_Pdf(file_bytes: bytes , thread_id: str , filename : Optional[str] = None) -> dict:
    """
    BUild a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """

    if not file_bytes:
        raise ValueError("NO bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False , suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000 , chunk_overlap = 200 , separators=["\n\n" , "\n" , " " ,""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks , embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity" , search_kwargs = {"k" : 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ***************************** Tools ****************************

search_tool = DuckDuckGoSearchRun(region="us-en")
semantic_scholar_tool = SemanticScholarQueryRun()

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=LUNW6OVK0TJQW581"
    r = requests.get(url)
    return r.json()

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python",
            "args": ["C:/Users/Parth/Desktop/mcp severs/main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        },
        "context7": {
            "transport": "streamable_http",
            "url": "https://mcp.context7.com/mcp"
        },
        "deepwiki": {
            "transport": "streamable_http",
            "url": "https://mcp.deepwiki.com/mcp"
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        tools = run_async(client.get_tools())
        print(f"Loaded MCP tools: {[t.name for t in tools]}")
        return tools
    except Exception as e:
        print(f"MCP tools failed to load: {e}")
        return []

mcp_tools = load_mcp_tools()

@tool
def rag_tool(query: str,thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    ALways Include the thread_id when calling this tool. 
    """

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return{
            "error" : "No Document indexed for this chat. Uploaded a PDF first",
            "query" : query
        }

    result = retriever.invoke(query)
    context = [doc.page_content for  doc in result]
    metadata =  [doc.metadata for doc in result]

    return {
        "query" : query,
        "context" : context,
        "metadata" : metadata,
        "source_file" : _THREAD_METADATA.get(str(thread_id) , {}).get("filename"),
    }

tools = [search_tool , semantic_scholar_tool , get_stock_price , *mcp_tools , rag_tool]
llm_with_tools = llm.bind_tools(tools) if tools else llm 


# **************************** State ***********************************

class ChatState(TypedDict):
    messages : Annotated[list[BaseModel] , add_messages]

# **************************** Nodes ***********************************    

async def chat_node(state: ChatState , config = None):
    """LLM node that may answer or request a tool call."""

    thread_id = None 
    if config and isinstance(config , dict):
        thread_id = config.get("configurable" , {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message , *state['messages']]
    try:
        response = await llm_with_tools.ainvoke(messages , config=config)
    except Exception as e:
        response = AIMessage(content=f"Sorry, I couldn't process that request. Please try rephrasing. (Error: {str(e)[:100]})")
    return {'messages' :[response] }

tool_node = ToolNode(tools, handle_tool_errors=True) if tools else None

# ************************* Checkpointer ******************************

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db" , check_same_thread = False)
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# ************************** Graph ***********************************

graph = StateGraph(ChatState)

graph.add_node('chat_node' , chat_node)
graph.add_edge(START , 'chat_node')

if tool_node:
    graph.add_node("tools" , tool_node)
    graph.add_conditional_edges('chat_node' , tools_condition)
    graph.add_edge('tools' , 'chat_node')
else:
    graph.add_edge("chat_node" , END)

chatbot = graph.compile(checkpointer=checkpointer)

# ********************************** Helper ****************************

async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})