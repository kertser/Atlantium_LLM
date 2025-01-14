# Declare and use BUILD_TYPE argument for subsequent stages
ARG BUILD_TYPE

# Base image
FROM python:3.12-slim AS base

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
        pciutils \
        sudo \
        libjson-c5 \
        patchelf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LD_LIBRARY_PATH=/app/models/agents/resources

# Create non-root user and setup sudo
RUN useradd -m -u 1000 appuser && \
    echo "appuser ALL=(ALL) NOPASSWD: /usr/bin/chown" >> /etc/sudoers

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

# Switch to root for setup
USER root

# Copy entrypoint script first and set its permissions
COPY scripts/docker-entrypoint.sh /app/
RUN chmod 755 /app/docker-entrypoint.sh

# Copy application code
COPY . .

# Create directories and set permissions in a single layer
RUN mkdir -p "/app/RAG_Data/stored_images" \
             "/app/RAG_Data/stored_text_chunks" \
             "/app/Raw Documents" \
             /app/logs \
             /app/models/agents/resources && \
    cp $(find /lib/x86_64-linux-gnu -name "libjson-c.so*" | grep -E "libjson-c\.so\.[0-9]+$") /app/models/agents/resources/libjson-c.so.5 && \
    chmod +x /app/models/agents/resources/libjson-c.so.5 && \
    chown -R appuser:appuser /app && \
    find /app -type d -exec chmod 775 {} \; && \
    find /app -type f -exec chmod 664 {} \; && \
    chmod 755 /app/docker-entrypoint.sh && \
    # Verify the ownership
    ls -la /app/RAG_Data && \
    ls -la "/app/Raw Documents" && \
    ls -la /app/logs && \
    ls -la /app/models/agents/resources

# Fix library paths if needed
RUN cd /app/models/agents/resources && \
    if [ -f "libred_api.so.1.0" ]; then \
        patchelf --set-rpath '\$ORIGIN' libred_api.so.1.0 && \
        ln -sf libred_api.so.1.0 libred_api.so && \
        ln -sf libred_api.so.1.0 libred_api.so.1; \
    fi

# Switch to appuser for runtime
USER appuser

# Verify setup as appuser
RUN id && \
    ls -la /app/RAG_Data

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 9000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "run.py"]