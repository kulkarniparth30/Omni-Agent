import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.access")

class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Attaches a unique X-Request-ID to every incoming request.
    2. Measures total processing latency.
    3. Logs the HTTP access log with method, path, status code, and latency.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{process_time_ms:.2f}"

        logger.info(
            "%s %s - %d (%.2f ms) [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            process_time_ms,
            request_id
        )
        return response
