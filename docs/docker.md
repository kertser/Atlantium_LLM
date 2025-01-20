# Docker Configuration

## Overview

The system leverages Docker Compose with dual-profile configurations (CPU and GPU) and persistent volume management. The deployment process detects hardware capabilities and selects the appropriate profile automatically.

## Dockerfile Structure

### Base Image
- **Base Image**: `python:3.10-slim`
- **Working Directory**: `/app`
- **Installed Dependencies**:
  - `build-essential`
  - `python3-dev`
  - `netcat-traditional`
  - `pciutils`
  - `sudo`
- **Environment Setup**:
  - `PYTHONUNBUFFERED=1`
  - `KMP_DUPLICATE_LIB_OK=TRUE`
  - `PYTHONDONTWRITEBYTECODE=1`
  - `PYTHONIOENCODING=utf-8`

### Build Stages
- **GPU Stage**:
  - Sets `USE_CPU=0`.
  - Runs `install_requirements.sh`.
- **CPU Stage**:
  - Sets `USE_CPU=1`.
  - Runs `install_requirements.sh`.
- **Final Stage**:
  - Determined by `BUILD_TYPE` (defaults to `cpu`).

### Security Configuration
- Creates non-root user `appuser` with appropriate sudo permissions.
- Configures directories and permissions:
  - Creates required directories:
    - `/app/RAG_Data/stored_images`
    - `/app/RAG_Data/stored_text_chunks`
    - `/app/Raw Documents`
    - `/app/logs`
  - Sets permissions for directories (`775`) and files (`664`).

## Docker Compose Configuration

### GPU Profile (`web-app-gpu`)
- **Environment Variables**:
  - `PYTHONIOENCODING=utf-8`
  - `USE_CPU=0`
- **Build Context**:
  - `Dockerfile` with `BUILD_TYPE=gpu`.
- **GPU Resources**:
  - Driver: `nvidia`
  - Count: `all`
  - Capabilities: `[gpu]`

### CPU Profile (`web-app-cpu`)
- **Environment Variables**:
  - `PYTHONIOENCODING=utf-8`
  - `USE_CPU=1`
- **Build Context**:
  - `Dockerfile` with `BUILD_TYPE=cpu`.

### Volume Configuration
- **Bind Mounts**:
  - `Raw Documents` directory.
  - `RAG_Data` directory.
  - `logs` directory.
  - Docker socket access.
  - Read-only Docker configuration.

## Entry Point Script Functions

- **fix_directory_permissions**: Ensures correct directory permissions.
- **initialize_rag**: Initializes the RAG database if `INITIALIZE_RAG=true`.
- **setup_processed_files**: Configures `processed_files.json`.

## Required Environment Variables

- **General**:
  - `CONTAINER_NAME=atlantium_llm-web-app-1`
  - `PYTHONIOENCODING=utf-8`
  - `USE_CPU=0/1`
  - `INITIALIZE_RAG=false`
- **Build**:
  - `BUILD_TYPE=gpu/cpu`
  - `DOCKER_BUILDKIT=1`

## Common Operations

### Container Management
- **View Logs**:
  ```bash
  docker logs atlantium_llm-web-app-1 -f --tail=100
  ```
- **Remove All Containers**:
  ```bash
  docker rm -f $(docker ps -a -q)
  ```
- **System Cleanup**:
  ```bash
  docker system prune --all --volumes --force
  ```

### Volume Management
- **List Volumes**:
  ```bash
  docker volume ls
  ```
- **Inspect Volumes**:
  ```bash
  docker volume inspect [volume_name]
  ```

### Resource Monitoring
- **Container Stats**:
  ```bash
  docker stats atlantium_llm-web-app-1
  ```
- **GPU Status**:
  ```bash
  docker exec atlantium_llm-web-app-1 nvidia-smi
  ```

## Troubleshooting

### GPU Detection Issues
- **Verify NVIDIA Drivers**:
  ```bash
  nvidia-smi
  ```
- **Check Container Toolkit**:
  ```bash
  nvidia-container-cli info
  ```

### Volume Issues
- **Check Permissions**:
  ```bash
  ls -la /var/lib/docker/volumes/
  ```
- **Verify Mount Points**:
  ```bash
  docker inspect atlantium_llm-web-app-1
  ```

### Build Issues
- **Clean Cache**:
  ```bash
  docker builder prune
  ```
- **Force CPU Build**:
  ```bash
  export BUILD_TYPE=cpu && ./deploy.sh
  ```

## Related Documentation

- [Installation Guide](../docs/installation.md)
- [Technical Reference](../docs/technical-reference.md)
- [Update Service](../docs/update-service.md)

