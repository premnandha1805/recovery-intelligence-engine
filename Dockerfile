# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Create isolated virtualenv for runtime dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.12-slim AS runtime

# Create non-root system user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy application code and dataset
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser decision_engine/ /app/decision_engine/
COPY --chown=appuser:appuser ml/ /app/ml/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser policy/ /app/policy/

# Switch to non-root user
USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Healthcheck using Python stdlib (no external curl/wget dependencies)
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run FastAPI decision engine service as a single worker process
CMD ["uvicorn", "decision_engine.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
