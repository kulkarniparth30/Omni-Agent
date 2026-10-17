from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import get_settings

def get_chat_llm(temperature: float = 0.0) -> ChatGroq:
    """Factory to create a ChatGroq instance configured from application settings."""
    settings = get_settings()
    return ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=temperature,
        streaming=True,
        api_key=settings.GROQ_API_KEY
    )

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Factory to create a Google Generative AI Embeddings instance from application settings."""
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDINGS_MODEL,
        google_api_key=settings.GOOGLE_API_KEY
    )
