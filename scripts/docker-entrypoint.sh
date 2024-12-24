#!/bin/bash
set -e

# Function to verify directory exists and has correct permissions
verify_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory missing: $dir"
        return 1
    }
    if [ "$(stat -c '%u:%g' "$dir")" != "$(id -u appuser):$(id -g appuser)" ]; then
        echo "ERROR: Wrong ownership on: $dir"
        return 1
    }
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
        echo "Directory verification failed"
        exit 1
    }
done

# Handle processed_files.json
PROCESSED_FILE="/app/processed_files.json"
if [ ! -f "$PROCESSED_FILE" ]; then
    echo "Creating $PROCESSED_FILE"
    echo "{}" > "$PROCESSED_FILE"
fi

# Verify permissions were set correctly
echo "Directory structure verification complete"
ls -la /app/RAG_Data
ls -la "/app/Raw Documents"
ls -la /app/logs
ls -la "$PROCESSED_FILE"

# Execute the command passed to docker-entrypoint
echo "Starting application..."
exec "$@"