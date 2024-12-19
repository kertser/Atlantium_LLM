#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/kertser/Atlantium_LLM.git"
APP_DIR="/home/user/Projects/Atlantium_LLM"
LOG_FILE="/app/logs/updates/rag_update.log"
BACKUP_DIR="/home/user/backups/rag"
DATE=$(date +%Y%m%d_%H%M%S)

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Create required directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

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
    git reset --hard origin/main
else
    log "Cloning fresh repository"
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# Set permissions
if ! chown -R user:user "$APP_DIR"; then
    log "Failed to set ownership of $APP_DIR"
    exit 1
fi

if ! chmod -R 755 "$APP_DIR"; then
    log "Failed to set permissions on $APP_DIR"
    exit 1
fi

# Rebuild and restart service
if [ -f "docker-compose.yaml" ]; then
    log "Rebuilding service with fresh image"
    docker-compose build --no-cache || {
        log "Build failed"
        exit 1
    }

    log "Stopping existing service"
    docker-compose down

    log "Starting updated service"
    docker-compose up -d || {
        log "Service startup failed"
        exit 1
    }

    # Clean old images
    log "Cleaning old images"
    docker image prune -f
else
    log "Error: docker-compose.yaml not found"
    exit 1
fi

# Clean old backups (keep last 5)
log "Cleaning old backups"
cd "$BACKUP_DIR" && ls -t rag_backup_*.tar.gz | tail -n +6 | xargs -r rm

log "Update completed successfully"
exit 0