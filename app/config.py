from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings read from environment variables or .env file.
    All configuration across the entire app is centralized here.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Providers
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # LLM Models
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDINGS_MODEL: str = "models/gemini-embedding-001"

    # External APIs
    ALPHA_VANTAGE_API_KEY: Optional[str] = None

    # MCP Server Configurations
    MCP_ARITH_SCRIPT_PATH: Optional[str] = None
    MCP_EXPENSE_URL: str = "https://splendid-gold-dingo.fastmcp.app/mcp"
    MCP_CONTEXT7_URL: str = "https://mcp.context7.com/mcp"
    MCP_DEEPWIKI_URL: str = "https://mcp.deepwiki.com/mcp"

    # Storage Paths
    DATABASE_PATH: str = "chatbot.db"
    INDEXES_DIR: Path = Path("data/indexes")

    # API Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"

@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings factory.
    lru_cache ensures .env is read once instead of on every function call.
    """
    return Settings()
