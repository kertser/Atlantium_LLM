# Declare and use BUILD_TYPE argument for subsequent stages
ARG BUILD_TYPE

# Base image
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies including lspci for GPU detection
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        netcat-traditional \
        pciutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create all necessary directories with correct structure
RUN mkdir -p "/app/RAG_Data/stored_images" \
    "/app/RAG_Data/stored_text_chunks" \
    "/app/Raw Documents" \
    /app/logs

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

# Switch to root temporarily for permissions
USER root

# Copy entrypoint script first and set its permissions
COPY scripts/docker-entrypoint.sh /app/
RUN chmod 755 /app/docker-entrypoint.sh

# Copy application code
COPY . .

# Set all permissions correctly
RUN chown -R appuser:appuser /app && \
    find /app -type d -exec chmod 775 {} \; && \
    find /app -type f -exec chmod 664 {} \; && \
    chmod 755 /app/docker-entrypoint.sh && \
    # Ensure these directories exist with correct ownership
    mkdir -p "/app/RAG_Data/stored_images" \
            "/app/RAG_Data/stored_text_chunks" \
            "/app/Raw Documents" \
            /app/logs && \
    chown -R appuser:appuser "/app/RAG_Data" \
            "/app/Raw Documents" \
            /app/logs

# Switch to appuser for runtime
USER appuser

EXPOSE 9000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "run.py"]