"""
tests/test_api.py

Automated integration and unit tests for the ToolBot / Omni-Agent API layer.
Tests run in-memory using FastAPI TestClient with lifespan management and dependency overrides.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app
from app.api.deps import get_agent

@pytest.fixture(scope="module")
def client():
    """Module-level fixture ensuring FastAPI lifespan runs on startup and shutdown."""
    with TestClient(app) as c:
        yield c

# ----------------------------------------------------
# 1. Health Endpoint Test
# ----------------------------------------------------
def test_health_endpoint(client):
    """Verify GET /api/v1/health returns HTTP 200 and lists active tools."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "tools_count" in data
    assert isinstance(data["available_tools"], list)

# ----------------------------------------------------
# 2. Pydantic Request Validation (HTTP 422)
# ----------------------------------------------------
def test_chat_validation_error(client):
    """Verify sending invalid or missing fields triggers HTTP 422 Unprocessable Entity."""
    # Empty JSON body
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422

    # Missing thread_id
    response = client.post("/api/v1/chat", json={"message": "Hello"})
    assert response.status_code == 422

# ----------------------------------------------------
# 3. Unknown Thread Error (HTTP 404)
# ----------------------------------------------------
def test_unknown_thread_returns_404(client):
    """Verify requesting messages for a non-existent thread returns HTTP 404."""
    response = client.get("/api/v1/threads/non-existent-thread-uuid-99999/messages")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data or "detail" in data

# ----------------------------------------------------
# 4. Chat Execution with Mocked Agent (HTTP 200)
# ----------------------------------------------------
def test_chat_endpoint_with_mock_agent(client):
    """Verify POST /api/v1/chat works end-to-end with a mocked LangGraph agent."""
    class MockCompiledGraph:
        async def ainvoke(self, input_dict, config=None):
            return {
                "messages": [
                    HumanMessage(content=input_dict["messages"][0].content),
                    AIMessage(content="This is a deterministic mocked AI response.")
                ]
            }

    # Inject mock agent into FastAPI's dependency injection container
    app.dependency_overrides[get_agent] = lambda: MockCompiledGraph()

    try:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello from pytest unit test!",
                "thread_id": "pytest-unit-thread-1"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "This is a deterministic mocked AI response."
        assert data["thread_id"] == "pytest-unit-thread-1"
        assert "latency_ms" in data
    finally:
        # Clean up dependency override so other tests aren't affected
        app.dependency_overrides.pop(get_agent, None)

# ----------------------------------------------------
# 5. Document Validation Test (HTTP 400)
# ----------------------------------------------------
def test_document_upload_rejects_non_pdf(client):
    """Verify uploading a non-PDF file returns HTTP 400 Bad Request."""
    files = {"file": ("malicious.exe", b"not a pdf", "application/octet-stream")}
    response = client.post("/api/v1/threads/test-thread/documents", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
