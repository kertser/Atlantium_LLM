# Base image
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Create and switch to non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements first for better caching
COPY requirements.txt .

# GPU stage
FROM base as gpu
ENV USE_CPU=0
RUN pip install --no-cache-dir -r requirements.txt

# CPU stage
FROM base as cpu
ENV USE_CPU=1
RUN pip install --no-cache-dir -r requirements.txt

# Final stage - will be selected during build
FROM ${BUILD_TYPE:-cpu}

USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with correct permissions
RUN mkdir -p "RAG_Data/stored_images" "Raw Documents" logs && \
    chmod -R 755 "RAG_Data" "Raw Documents" logs

# Expose the port
EXPOSE 9000

CMD ["python", "run.py"]