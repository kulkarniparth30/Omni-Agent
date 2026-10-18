from contextlib import asynccontextmanager
import logging
import aiosqlite
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.middleware import RequestCorrelationMiddleware
from app.core.exceptions import ToolBotException
from app.tools import get_all_tools, load_mcp_tools
from app.agent.graph import build_graph
from app.services.pdf_store import list_persisted_threads
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import routes
from app.api.routes.health import router as health_router
from app.api.routes.chat import router as chat_router
from app.api.routes.threads import router as threads_router
from app.api.routes.documents import router as documents_router

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager:
    - Runs ONCE at server startup to connect SQLite, load MCP tools, and compile the agent.
    - Runs ONCE at server shutdown to cleanly close database pools and network clients.
    Replaces the custom background-thread event loop from the legacy script.
    """
    settings = get_settings()
    logger.info("Starting ToolBot API service (env: %s)...", settings.APP_ENV)

    # 1. Initialize persistent SQLite Checkpointer
    logger.info("Connecting to SQLite checkpointer database at '%s'...", settings.DATABASE_PATH)
    db_conn = await aiosqlite.connect(database=settings.DATABASE_PATH, check_same_thread=False)
    checkpointer = AsyncSqliteSaver(db_conn)

    # 2. Asynchronously load external MCP tools
    logger.info("Discovering and loading MCP tools...")
    mcp_tools = await load_mcp_tools()

    # 3. Register all tools
    all_tools = get_all_tools(mcp_tools=mcp_tools)
    logger.info("Registered %d tools: %s", len(all_tools), [t.name for t in all_tools])

    # 4. Compile the LangGraph conversation graph
    logger.info("Compiling LangGraph conversation engine...")
    agent = build_graph(tools=all_tools, checkpointer=checkpointer)

    # 5. Check for persisted PDF indexes
    persisted_docs = list_persisted_threads()
    logger.info("Found %d persisted PDF document index(es) on disk: %s", len(persisted_docs), persisted_docs)

    # 6. Attach shared resources to app.state
    app.state.db_conn = db_conn
    app.state.checkpointer = checkpointer
    app.state.tools = all_tools
    app.state.agent = agent

    logger.info("ToolBot API is ready to accept requests.")
    yield  # Application is live and serving HTTP requests

    # --- SHUTDOWN CLEANUP ---
    logger.info("Shutting down ToolBot API service...")
    if hasattr(app.state, "db_conn") and app.state.db_conn:
        await app.state.db_conn.close()
        logger.info("SQLite connection closed.")

def create_application() -> FastAPI:
    """Factory function to instantiate the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ToolBot API",
        description="Production API for ToolBot Multi-Utility AI Agent",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration (allows Streamlit UI and local clients to connect)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request correlation & latency logging middleware
    app.add_middleware(RequestCorrelationMiddleware)

    # Global custom domain exception handler
    @app.exception_handler(ToolBotException)
    async def toolbot_exception_handler(request: Request, exc: ToolBotException):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.warning("Domain exception: %s (code=%d, req_id=%s)", exc.message, exc.status_code, req_id)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
                "request_id": req_id
            }
        )

    # Unhandled server errors handler
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error("Unhandled server exception: %s (req_id=%s)", str(exc), req_id, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error occurred.",
                "details": str(exc),
                "request_id": req_id
            }
        )

    # Register routers with API prefix
    api_prefix = settings.API_V1_PREFIX
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(chat_router, prefix=api_prefix)
    app.include_router(threads_router, prefix=api_prefix)
    app.include_router(documents_router, prefix=api_prefix)

    # Also register /health at root for cloud health checkers
    app.include_router(health_router)

    return app

app = create_application()
