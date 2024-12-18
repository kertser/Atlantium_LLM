# Base image
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies including lspci for GPU detection
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        netcat-traditional \
        pciutils \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    DOCKER_CONTAINER=1

# Create and switch to non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

# Copy requirements files and installation script
COPY requirements_cpu.txt requirements_gpu.txt install_requirements.sh ./

# Set execute permissions on the script
RUN chmod +x install_requirements.sh

# GPU stage
FROM base as gpu
ENV USE_CPU=0
RUN pip install --upgrade pip
RUN ./install_requirements.sh

# CPU stage
FROM base as cpu
ENV USE_CPU=1
RUN pip install --upgrade pip
RUN ./install_requirements.sh

# Final stage - will be selected during build
FROM ${BUILD_TYPE:-cpu}

USER root
# Copy application code and set permissions
COPY --chown=appuser:appuser . .
RUN chmod +x docker-entrypoint.sh

# Create necessary directories with correct permissions
RUN mkdir -p "RAG_Data/stored_images" && \
    mkdir -p "Raw Documents" && \
    mkdir -p logs/updates && \
    touch logs/updates/rag_update.log && \
    chown -R appuser:appuser "RAG_Data" && \
    chown -R appuser:appuser "Raw Documents" && \
    chown -R appuser:appuser logs && \
    chmod -R 755 "RAG_Data" && \
    chmod -R 755 "Raw Documents" && \
    chmod -R 755 logs && \
    chmod 644 logs/updates/rag_update.log

USER appuser

# Expose the port
EXPOSE 9000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "run.py"]