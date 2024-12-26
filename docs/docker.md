# Docker Configuration

## Overview

The system uses Docker Compose with dual profile configuration (CPU/GPU) and persistent volume management. The deployment process automatically detects hardware capabilities and selects the appropriate profile.

## Dockerfile Structure

### Base Image
```dockerfile
ARG BUILD_TYPE

FROM python:3.10-slim AS base
WORKDIR /app

# System dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    netcat-traditional \
    pciutils \
    sudo

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8
```

### Stage Selection
```dockerfile
# GPU stage
FROM base AS gpu
ENV USE_CPU=0
RUN ./install_requirements.sh

# CPU stage
FROM base AS cpu
ENV USE_CPU=1
RUN ./install_requirements.sh

# Final stage
FROM ${BUILD_TYPE:-cpu}
```

### Security Configuration
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser && \
    echo "appuser ALL=(ALL) NOPASSWD: /usr/bin/chown" >> /etc/sudoers

# Set permissions
RUN mkdir -p "/app/RAG_Data/stored_images" \
             "/app/RAG_Data/stored_text_chunks" \
             "/app/Raw Documents" \
             /app/logs && \
    chown -R appuser:appuser /app && \
    find /app -type d -exec chmod 775 {} \; && \
    find /app -type f -exec chmod 664 {} \;
```

## Docker Compose Configuration

### GPU Profile
```yaml
web-app-gpu:
    container_name: atlantium_llm-web-app-1
    environment:
        - PYTHONIOENCODING=utf-8
        - USE_CPU=0
    build:
        context: .
        dockerfile: Dockerfile
        args:
            - BUILD_TYPE=gpu
    deploy:
        resources:
            reservations:
                devices:
                    - driver: nvidia
                      count: all
                      capabilities: [gpu]
```

### CPU Profile
```yaml
web-app-cpu:
    container_name: atlantium_llm-web-app-1
    environment:
        - PYTHONIOENCODING=utf-8
        - USE_CPU=1
    build:
        context: .
        dockerfile: Dockerfile
        args:
            - BUILD_TYPE=cpu
```

### Volume Configuration
```yaml
volumes:
    - type: bind
      source: ./Raw Documents
      target: /app/Raw Documents
    - type: bind
      source: ./RAG_Data
      target: /app/RAG_Data
    - type: bind
      source: ./logs
      target: /app/logs
    - /var/run/docker.sock:/var/run/docker.sock
    - ${HOME}/.docker/config.json:/root/.docker/config.json:ro
```

## Entry Point Script

### Main Functions
```bash
# Directory setup and permissions
fix_directory_permissions() {
    # Set correct permissions for directories
}

# Initialize RAG database
initialize_rag() {
    # Initialize if INITIALIZE_RAG=true
}

# Handle processed_files.json
setup_processed_files() {
    # Create and configure processed_files.json
}
```

## Environment Variables

### Required Variables
```bash
# Container configuration
CONTAINER_NAME=atlantium_llm-web-app-1
PYTHONIOENCODING=utf-8
USE_CPU=0/1
INITIALIZE_RAG=false

# Build configuration
BUILD_TYPE=gpu/cpu
DOCKER_BUILDKIT=1
```

## Common Operations

### Container Management
```bash
# View container logs
docker logs atlantium_llm-web-app-1 -f --tail=100

# Remove all containers
docker rm -f $(docker ps -a -q)

# System cleanup
docker system prune --all --volumes --force
```

### Volume Management
```bash
# List volumes
docker volume ls

# Inspect volumes
docker volume inspect raw_docs
docker volume inspect rag_data
```

### Resource Monitoring
```bash
# Monitor container resources
docker stats atlantium_llm-web-app-1

# Check GPU status (GPU profile)
docker exec atlantium_llm-web-app-1 nvidia-smi
```

## Troubleshooting

### Common Issues

1. GPU Detection Failures
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check container toolkit
nvidia-container-cli info
```

2. Volume Persistence
```bash
# Check volume permissions
ls -la /var/lib/docker/volumes/

# Verify mount points
docker inspect atlantium_llm-web-app-1
```

3. Build Failures
```bash
# Clean build cache
docker builder prune

# Force CPU profile
export BUILD_TYPE=cpu && ./deploy.sh
```

## Related Documentation

- [Installation Guide](../docs/installation.md)
- [Technical Reference](../docs/technical-reference.md)
- [Update Service](../docs/update-service.md)