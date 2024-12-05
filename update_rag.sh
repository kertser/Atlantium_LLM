# sudo chmod +x /usr/local/bin/update_rag.sh
#!/bin/bash

# Log file
LOG_FILE="/var/log/rag_update.log"

# Function for logging
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Change to project directory
cd ~/Projects/Atlantium_LLM || exit 1

# Log start
log "Starting update process"

# Pull latest changes
git pull origin main

# Check if docker-compose exists and restart services
if [ -f "docker-compose.yaml" ]; then
    log "Rebuilding and restarting Docker services"
    docker-compose build
    docker-compose up -d
else
    log "Error: docker-compose.yaml not found"
    exit 1
fi

log "Update completed successfully"