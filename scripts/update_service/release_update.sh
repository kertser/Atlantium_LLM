#!/bin/bash

# release_update.sh
#
# Purpose: Automatically update Atlantium RAG application from GitHub release branch
# Author: Your Name
# Date: December 2024
#
# This script:
# 1. Monitors the release branch for new tagged versions
# 2. Creates backups before updates
# 3. Updates the application code
# 4. Rebuilds and restarts Docker containers
# 5. Verifies the update success
#
# Requirements:
# - Docker and docker-compose installed
# - Git installed
# - User running script must be in docker group
# - Application directory structure set up

#!/bin/bash

# release_update.sh
# Updates Atlantium RAG system from GitHub release branch while preserving Docker volumes

set -e

# Detect user environment
CURRENT_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$CURRENT_USER)
APP_DIR=${APP_DIR:-"$USER_HOME/Projects/Atlantium_LLM"}
LOG_DIR="$APP_DIR/logs/updates"
BACKUP_DIR="$APP_DIR/backups"
CONTAINER_NAME=${CONTAINER_NAME:-"atlantium_llm-web-app-1"}

# Create required directories
mkdir -p "$LOG_DIR"
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/update.log"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Function to check docker
check_docker() {
    log "Checking Docker status..."
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or user doesn't have docker permissions"
    fi
}

# Function to detect GPU configuration (from your deploy.sh)
detect_gpu_configuration() {
    log "Checking GPU configuration..."
    
    # Check for NVIDIA drivers
    if [ -f "/proc/driver/nvidia/version" ] && nvidia-smi &> /dev/null; then
        if command -v nvidia-container-cli &> /dev/null; then
            log "GPU configuration verified successfully"
            return 0
        fi
    fi
    
    log "Using CPU configuration"
    return 1
}

# Function to verify installation
verify_installation() {
    log "Verifying installation..."
    if [ ! -d "$APP_DIR" ]; then
        error_exit "Application directory not found: $APP_DIR"
    fi
    
    if [ ! -f "$APP_DIR/docker-compose.yaml" ]; then
        error_exit "docker-compose.yaml not found in $APP_DIR"
    }
}

# Function to create backup (excluding Docker volumes)
create_backup() {
    log "Creating backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/backup_$timestamp.tar.gz"
    
    # Backup code and configuration, excluding Docker volumes and temporary files
    tar --exclude='*.log' \
        --exclude='*.tmp' \
        --exclude='RAG_Data' \
        --exclude='Raw Documents' \
        --exclude="$BACKUP_DIR" \
        -czf "$BACKUP_FILE" -C "$APP_DIR" . || error_exit "Backup failed"
    
    # Clean old backups
    cd "$BACKUP_DIR" || error_exit "Cannot access backup directory"
    ls -t backup_*.tar.gz | tail -n +6 | xargs -r rm
    
    log "Backup created: $BACKUP_FILE"
}

# Function to update code
update_code() {
    log "Updating code..."
    cd "$APP_DIR" || error_exit "Cannot access application directory"
    
    # Get current version
    OLD_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
    
    # Fetch latest changes
    git fetch origin || error_exit "Failed to fetch from repository"
    
    # Switch to release branch
    git checkout release || error_exit "Failed to checkout release branch"
    git pull origin release || error_exit "Failed to pull release branch"
    
    # Get latest release tag
    LATEST_TAG=$(git describe --tags --abbrev=0 origin/release 2>/dev/null)
    
    if [ -z "$LATEST_TAG" ]; then
        error_exit "No release tags found on release branch"
    fi
    
    # Check if update is needed
    if [ "$OLD_VERSION" = "$LATEST_TAG" ]; then
        log "Already at latest version: $LATEST_TAG"
        exit 0
    fi
    
    log "Updating from $OLD_VERSION to $LATEST_TAG"
    
    # Checkout latest tag
    git checkout "$LATEST_TAG" || error_exit "Failed to checkout latest tag"
}

# Function to update docker container
update_docker() {
    log "Updating docker container..."
    cd "$APP_DIR" || error_exit "Cannot access application directory"
    
    # Detect GPU/CPU configuration
    if detect_gpu_configuration; then
        export BUILD_TYPE=gpu
        export USE_CPU=0
        PROFILE="gpu"
    else
        export BUILD_TYPE=cpu
        export USE_CPU=1
        PROFILE="cpu"
    fi

    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    
    # Stop current container (preserving volumes)
    log "Stopping current container..."
    docker-compose down || log "Warning: Issue stopping containers"
    
    # Clean old images
    log "Cleaning old images..."
    docker image prune -f
    
    # Build new container
    log "Building new container..."
    if ! docker-compose build --no-cache; then
        log "Build failed. Retrying with CPU configuration."
        export BUILD_TYPE=cpu
        export USE_CPU=1
        PROFILE="cpu"
        if ! docker-compose build --no-cache; then
            error_exit "Build failed even with CPU configuration"
        fi
    fi
    
    # Start new container
    log "Starting new container..."
    if ! docker-compose --profile ${PROFILE} up -d; then
        error_exit "Failed to start container"
    fi
    
    # Wait for container
    log "Waiting for container to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:9000 >/dev/null; then
            log "Container is ready"
            return 0
        fi
        sleep 2
    done
    
    error_exit "Container failed to become ready"
}

# Function to verify update
verify_update() {
    log "Verifying update..."
    
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        error_exit "Container is not running after update"
    fi
    
    if ! curl -s http://localhost:9000 >/dev/null; then
        error_exit "Service is not responding after update"
    fi
    
    # Check container logs for errors
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -i "error"; then
        log "Warning: Found errors in container logs"
    fi
    
    log "Update verification successful"
}

# Main update process
main() {
    log "Starting update process..."
    log "Using APP_DIR: $APP_DIR"
    log "Running as user: $CURRENT_USER"
    
    check_docker
    verify_installation
    create_backup
    update_code
    update_docker
    verify_update
    
    log "Update completed successfully"
}

# Run main function
main