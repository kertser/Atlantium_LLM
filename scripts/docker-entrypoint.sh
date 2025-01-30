#!/bin/bash
set -e

# Function to fix directory permissions if needed
fix_directory_permissions() {
    local dir="$1"
    echo "Checking permissions for: $dir"

    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir" || {
            echo "Failed to create directory: $dir"
            return 1
        }
    fi

    current_owner=$(stat -c '%u:%g' "$dir")
    if [ "$current_owner" != "1000:1000" ]; then
        echo "Fixing ownership of $dir"
        if command -v sudo >/dev/null 2>&1; then
            sudo chown -R appuser:appuser "$dir" || {
                echo "Failed to change ownership of $dir"
                return 1
            }
        else
            echo "Warning: sudo not available, cannot fix permissions"
            return 1
        fi
    fi

    echo "Directory $dir is now owned by: $(stat -c '%u:%g' "$dir")"
}

# Debug information
echo "Initial state:"
id
echo "Directory permissions before fixes:"
ls -la /app/RAG_Data

# Fix permissions for all required directories
echo "Fixing permissions for all directories..."
for dir in "/app/RAG_Data" "/app/RAG_Data/stored_images" "/app/RAG_Data/stored_text_chunks" "/app/Raw Documents" "/app/logs"; do
    fix_directory_permissions "$dir" || {
        echo "Failed to fix permissions for $dir"
        exit 1
    }
done

# Initialize RAG database if requested
if [ "$INITIALIZE_RAG" = "true" ]; then
    echo "Initializing RAG database..."
    python -c "from utils.initialize_RAG import initialize_rag_database; initialize_rag_database()" || {
        echo "Failed to initialize RAG database"
        exit 1
    }
fi

# Handle processed_files.json
PROCESSED_FILE="/app/RAG_Data/processed_files.json"
if [ ! -f "$PROCESSED_FILE" ]; then
    echo "Creating $PROCESSED_FILE"
    echo "{}" > "$PROCESSED_FILE" || {
        echo "Failed to create $PROCESSED_FILE"
        exit 1
    }
    if command -v sudo >/dev/null 2>&1; then
        sudo chown appuser:appuser "$PROCESSED_FILE" || {
            echo "Failed to change ownership of $PROCESSED_FILE"
            exit 1
        }
    fi
    chmod 664 "$PROCESSED_FILE" || {
        echo "Failed to set permissions on $PROCESSED_FILE"
        exit 1
    }
else
    echo "$PROCESSED_FILE already exists, skipping creation."
fi

echo "Final directory structure:"
ls -la /app/RAG_Data
ls -la "/app/Raw Documents"
ls -la /app/logs
ls -la "$PROCESSED_FILE"

echo "Starting application..."
exec "$@"