class ToolBotException(Exception):
    """Base exception for all domain errors within ToolBot."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ThreadNotFoundError(ToolBotException):
    """Raised when a requested thread ID does not exist in checkpoints."""
    def __init__(self, thread_id: str):
        super().__init__(
            message=f"Conversation thread '{thread_id}' not found.",
            status_code=404,
            details={"thread_id": thread_id}
        )

class LLMExecutionError(ToolBotException):
    """Raised when the downstream LLM provider (e.g. Groq) fails or times out."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=f"LLM generation failed: {message}",
            status_code=502,
            details=details or {}
        )

class DocumentProcessingError(ToolBotException):
    """Raised when parsing or indexing an uploaded PDF fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=f"Failed to process document: {message}",
            status_code=422,
            details=details or {}
        )
