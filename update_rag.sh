#!/bin/bash
set -e

# Use environment variables with defaults for flexibility
REPO_URL=${REPO_URL:-"https://github.com/kertser/Atlantium_LLM.git"}
BASE_DIR=${BASE_DIR:-"$HOME/Projects"}
APP_DIR="${BASE_DIR}/Atlantium_LLM"
LOG_DIR="${APP_DIR}/logs/updates"
BACKUP_DIR="${HOME}/backups/rag"

# Ensure we're running with adequate permissions
if [ "$EUID" -ne 0 ]; then
    if ! groups | grep -q docker; then
        echo "Please run as root or ensure user is in docker group"
        exit 1
    fi
fi

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Create required directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

# Ensure proper permissions
chown -R ${ACTUAL_USER}:${ACTUAL_USER} "$LOG_DIR"
chown ${ACTUAL_USER}:${ACTUAL_USER} "$LOG_FILE"

# Error handling
trap 'log "Error occurred. Exit code: $?"' ERR

log "Starting RAG system update process"

# Create backup
if [ -d "$APP_DIR" ]; then
    log "Creating backup of current state"
    tar -czf "$BACKUP_DIR/rag_backup_$DATE.tar.gz" -C "$(dirname "$APP_DIR")" "$(basename "$APP_DIR")" || {
        log "Backup failed but continuing with update"
    }
fi

# Update repository
if [ -d "$APP_DIR/.git" ]; then
    log "Updating existing repository"
    cd "$APP_DIR"
    git fetch origin
    git reset --hard origin/master
else
    log "Cloning fresh repository"
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# Set permissions
chown -R ${ACTUAL_USER}:${ACTUAL_USER} "$APP_DIR"
chmod -R 755 "$APP_DIR"

# Rebuild and restart service with proper cleanup
if [ -f "docker-compose.yaml" ]; then
    log "Stopping existing service"
    docker-compose down -v  # Stop and remove volumes

    log "Cleaning old images"
    docker image prune -f

    log "Rebuilding service with fresh image"
    DOCKER_BUILDKIT=1 docker-compose build --no-cache || {
        log "Build failed"
        exit 1
    }

    log "Starting updated service"
    docker-compose --profile cpu up -d --force-recreate || {
        log "Service startup failed"
        exit 1
    }

    # Verify service is running
    sleep 5
    if ! docker ps | grep -q "atlantium_llm-web-app"; then
        log "Service failed to start properly"
        docker-compose logs
        exit 1
    fi
else
    log "Error: docker-compose.yaml not found"
    exit 1
fi

# Clean old backups (keep last 5)
log "Cleaning old backups"
cd "$BACKUP_DIR" && ls -t rag_backup_*.tar.gz | tail -n +6 | xargs -r rm

log "Update completed successfully"
exit 0