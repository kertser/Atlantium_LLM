#!/bin/bash

# release_update.sh
#
# Purpose: Automatically update Atlantium RAG application from GitHub release branch
# Author: Your Name
# Date: December 2024

set -e

# Detect user environment
SERVICE_VERSION="1.0.0"
CURRENT_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~"$CURRENT_USER")
APP_DIR=${APP_DIR:-"$USER_HOME/Projects/Atlantium_LLM"}
LOG_DIR="$APP_DIR/logs/updates"
BACKUP_DIR="$APP_DIR/backups"
CONTAINER_NAME=${CONTAINER_NAME:-"atlantium_llm-web-app-1"}
TEMP_DIR="/tmp/atlantium_update_${RANDOM}"

# Logging function with proper file handling
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$timestamp - $1"
    echo "$message"
    # Use temporary file to avoid race conditions
    echo "$message" > "${TEMP_DIR}/temp.log" && \
    cat "${TEMP_DIR}/temp.log" >> "$LOG_DIR/update.log"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    rm -rf "${TEMP_DIR}"
    exit 1
}

# Cleanup function
cleanup() {
    rm -rf "${TEMP_DIR}"
}

# Set up temporary directory and trap
setup_temp() {
    mkdir -p "${TEMP_DIR}" || error_exit "Cannot create temporary directory"
    trap cleanup EXIT
}

# Create required directories with proper permissions
setup_directories() {
    for dir in "$LOG_DIR" "$BACKUP_DIR"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir" || error_exit "Cannot create directory: $dir"
            chmod 775 "$dir" || error_exit "Cannot set permissions on: $dir"
        fi
    done
}

# Function to check docker
check_docker() {
    log "Checking Docker status..."
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or user doesn't have docker permissions"
    fi
}

# Function to verify installation
verify_installation() {
    log "Verifying installation..."
    if [ ! -d "$APP_DIR" ]; then
        error_exit "Application directory not found: $APP_DIR"
    fi
    if [ ! -f "$APP_DIR/docker-compose.yaml" ]; then
        error_exit "docker-compose.yaml not found in $APP_DIR"
    fi

    # Verify write permissions
    if ! touch "$LOG_DIR/.write_test" 2>/dev/null; then
        error_exit "Cannot write to log directory"
    fi
    rm -f "$LOG_DIR/.write_test"

    if ! touch "$BACKUP_DIR/.write_test" 2>/dev/null; then
        error_exit "Cannot write to backup directory"
    fi
    rm -f "$BACKUP_DIR/.write_test"
}

# Function to create backup with proper error handling
create_backup() {
    log "Creating backup..."
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local temp_backup="${TEMP_DIR}/backup_${timestamp}.tar.gz"
    local final_backup="$BACKUP_DIR/backup_${timestamp}.tar.gz"

    # Create backup in temporary location first
    if ! tar --exclude='*.log' \
            --exclude='*.tmp' \
            --exclude='RAG_Data' \
            --exclude='Raw Documents' \
            --exclude="$BACKUP_DIR" \
            --exclude="${TEMP_DIR}" \
            -czf "$temp_backup" -C "$APP_DIR" .; then
        error_exit "Failed to create backup"
    fi

    # Move backup to final location
    if ! mv "$temp_backup" "$final_backup"; then
        error_exit "Failed to move backup to final location"
    fi

    # Clean old backups (keep last 5)
    find "$BACKUP_DIR" -name "backup_*.tar.gz" -type f -printf '%T@ %p\n' | \
        sort -n | head -n -5 | cut -d' ' -f2- | xargs -r rm

    log "Backup created successfully: $final_backup"
}

# Function to detect GPU configuration
detect_gpu_configuration() {
    log "Checking GPU configuration..."
    if [ -f "/proc/driver/nvidia/version" ] && nvidia-smi &> /dev/null; then
        if command -v nvidia-container-cli &> /dev/null; then
            log "GPU configuration verified successfully"
            return 0
        fi
    fi
    log "Using CPU configuration"
    return 1
}

# Function to update code
update_code() {
    log "Updating code..."
    cd "$APP_DIR" || error_exit "Cannot access application directory"

    OLD_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")

    if ! git fetch origin; then
        error_exit "Failed to fetch from repository"
    fi

    if ! git checkout release; then
        error_exit "Failed to checkout release branch"
    fi

    if ! git pull origin release; then
        error_exit "Failed to pull release branch"
    fi

    LATEST_TAG=$(git describe --tags --abbrev=0 origin/release 2>/dev/null)

    if [ -z "$LATEST_TAG" ]; then
        error_exit "No release tags found"
    fi

    if [ "$OLD_VERSION" = "$LATEST_TAG" ]; then
        log "Already at latest version: $LATEST_TAG"
        exit 0
    fi

    log "Updating from $OLD_VERSION to $LATEST_TAG"

    if ! git checkout "$LATEST_TAG"; then
        error_exit "Failed to checkout latest tag"
    fi
}

# Function to update docker container
update_docker() {
    log "Updating docker container..."
    cd "$APP_DIR" || error_exit "Cannot access application directory"

    if detect_gpu_configuration; then
        export BUILD_TYPE=gpu
        export USE_CPU=0
        PROFILE="gpu"
    else
        export BUILD_TYPE=cpu
        export USE_CPU=1
        PROFILE="cpu"
    fi

    export DOCKER_BUILDKIT=1

    log "Stopping current container..."
    docker-compose down || log "Warning: Issue stopping containers"

    log "Cleaning old images..."
    docker image prune -f

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

    log "Starting new container..."
    if ! docker-compose --profile ${PROFILE} up -d; then
        error_exit "Failed to start container"
    fi

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

    if docker logs "$CONTAINER_NAME" 2>&1 | grep -i "error"; then
        log "Warning: Found errors in container logs"
    fi

    log "Update verification successful"
}

# Main update process
main() {
    setup_temp
    setup_directories

    log "Update service version: $SERVICE_VERSION"
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