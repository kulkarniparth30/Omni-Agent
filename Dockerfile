# ==============================================================================
# STAGE 1: Build & Dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation essentials for packages that require C++ (e.g. FAISS, C-bindings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a separate wheels cache or user directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==============================================================================
# STAGE 2: Lean Production Runtime
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Set production environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production

# Install minimal runtime dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled dependencies from builder stage
COPY --from=builder /install /usr/local

# Create non-root application user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Create necessary directories and set ownership
RUN mkdir -p /app/data/indexes && chown -R appuser:appgroup /app

# Copy application source code
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup ui/ /app/ui/
COPY --chown=appuser:appgroup evals/ /app/evals/

# Switch to unprivileged user
USER appuser

# Expose standard FastAPI port
EXPOSE 8000

# Docker healthcheck pinging the FastAPI health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
