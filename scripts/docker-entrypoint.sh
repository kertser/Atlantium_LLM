#!/bin/bash
set -e

# Function to verify directory exists and has correct permissions
verify_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory missing: $dir"
        return 1
    fi

    # Get numeric UID and GID of appuser
    local app_uid=$(id -u appuser)
    local app_gid=$(id -g appuser)

    # Get ownership of directory
    local dir_owner=$(stat -c '%u' "$dir")
    local dir_group=$(stat -c '%g' "$dir")

    if [ "$dir_owner" != "$app_uid" ] || [ "$dir_group" != "$app_gid" ]; then
        echo "ERROR: Wrong ownership on: $dir"
        echo "Expected $app_uid:$app_gid, got $dir_owner:$dir_group"
        return 1
    fi

    echo "Verified directory: $dir"
    return 0
}

# Initialize RAG database if requested
if [ "$INITIALIZE_RAG" = "true" ]; then
    echo "Initializing RAG database..."
    python -c "from utils.initialize_RAG import initialize_rag_database; initialize_rag_database()" || {
        echo "Failed to initialize RAG database"
        exit 1
    }
fi

# Debug information
echo "Current user: $(id)"
echo "appuser details: $(id appuser)"
echo "Directory ownership:"
ls -la /app/RAG_Data

# Verify required directories
required_dirs=(
    "/app/RAG_Data"
    "/app/RAG_Data/stored_images"
    "/app/RAG_Data/stored_text_chunks"
    "/app/Raw Documents"
    "/app/logs"
)

echo "Verifying directory structure..."
for dir in "${required_dirs[@]}"; do
    verify_directory "$dir" || {
        echo "Directory verification failed for: $dir"
        exit 1
    }
done

# Handle processed_files.json
PROCESSED_FILE="/app/processed_files.json"
if [ ! -f "$PROCESSED_FILE" ]; then
    echo "Creating $PROCESSED_FILE"
    echo "{}" > "$PROCESSED_FILE"
    chown appuser:appuser "$PROCESSED_FILE"
    chmod 664 "$PROCESSED_FILE"
fi

# Verify final structure
echo "Directory structure verification complete"
ls -la /app/RAG_Data
ls -la "/app/Raw Documents"
ls -la /app/logs
ls -la "$PROCESSED_FILE"

# Execute the command passed to docker-entrypoint
echo "Starting application..."
exec "$@"