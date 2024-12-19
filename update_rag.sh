#!/bin/bash

# Configuration
REPO_URL="https://github.com/kertser/Atlantium_LLM.git"
APP_DIR="/home/user/Projects/Atlantium_LLM"
LOG_FILE="/app/logs/updates/rag_update.log"  # Updated path
BACKUP_DIR="/home/user/backups/rag"
DATE=$(date +%Y%m%d_%H%M%S)

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"  # Always print to stdout
    if [ -w "/var/log/rag_update.log" ]; then    # Only write to file if writable
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "/var/log/rag_update.log"
    else
        echo "Warning: Cannot write to log file"
    fi
}

# Error handling
set -e
trap 'log "Error occurred. Exit code: $?"' ERR

# Create directories if they don't exist
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Backup current state
log "Creating backup..."
if [ -d "$APP_DIR" ]; then
    tar -czf "$BACKUP_DIR/rag_backup_$DATE.tar.gz" -C "$(dirname "$APP_DIR")" "$(basename "$APP_DIR")"
fi

# Update application
log "Starting update process..."

# Check if directory exists and is a git repo
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR"
    git fetch origin
    git reset --hard origin/main
else
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

if ! chown -R ubuntu:ubuntu "$APP_DIR"; then
    log "Failed to set ownership of $APP_DIR"
    exit 1
fi
if ! chmod -R 755 "$APP_DIR"; then
    log "Failed to set permissions on $APP_DIR"
    exit 1
fi

# Restart Docker service
if [ -f "docker-compose.yaml" ]; then
    log "Rebuilding and restarting service..."

    # Build with no cache to ensure fresh build
    docker-compose build --no-cache

    # Restart service
    docker-compose down
    docker-compose up -d

    # Clean up old images
    docker image prune -f
else
    log "Error: docker-compose.yaml not found"
    exit 1
fi

# Clean old backups (keep last 5)
log "Cleaning old backups..."
cd "$BACKUP_DIR" && ls -t rag_backup_*.tar.gz | tail -n +6 | xargs -r rm

log "Update completed successfully"