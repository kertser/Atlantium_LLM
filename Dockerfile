# Base image
FROM python:3.10-slim AS base

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
    PYTHONIOENCODING=utf-8

# Create and switch to non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

# Copy requirements files and installation script
COPY requirements_cpu.txt requirements_gpu.txt scripts/install_requirements.sh ./

# Set execute permissions on the script
RUN chmod +x install_requirements.sh

# GPU stage
FROM base AS gpu
ENV USE_CPU=0
RUN ./install_requirements.sh

# CPU stage
FROM base AS cpu
ENV USE_CPU=1
RUN ./install_requirements.sh

# Final stage - will be selected during build
FROM ${BUILD_TYPE:-cpu}

USER root
# Copy application code and set permissions
COPY --chown=appuser:appuser . .
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Create necessary directories with correct permissions
RUN mkdir -p "RAG_Data/stored_images" "Raw Documents" logs \
    && chown -R appuser:appuser "RAG_Data" "Raw Documents" logs \
    && chmod -R 755 "RAG_Data" "Raw Documents" logs

USER appuser

# Expose the port
EXPOSE 9000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "run.py"]