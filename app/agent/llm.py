from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import get_settings

def get_chat_llm(temperature: float = 0.0) -> ChatGroq:
    """Factory to create a ChatGroq instance configured from application settings."""
    settings = get_settings()
    api_key = settings.GROQ_API_KEY or "gsk_placeholder_key"
    return ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=temperature,
        streaming=True,
        api_key=api_key
    )

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Factory to create a Google Generative AI Embeddings instance from application settings."""
    settings = get_settings()
    google_key = settings.GOOGLE_API_KEY or "placeholder_google_key"
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDINGS_MODEL,
        google_api_key=google_key
    )
