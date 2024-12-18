#!/bin/bash
set -e

# Initialize RAG database if requested
if [ "$INITIALIZE_RAG" = "true" ]; then
    echo "Initializing RAG database..."
    python -c "from utils.initialize_RAG import initialize_rag_database; initialize_rag_database()"
fi

# Create required directories if they don't exist
for dir in "RAG_Data/stored_images" "Raw Documents" logs; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

# Set proper permissions
chmod -R 755 "RAG_Data" "Raw Documents" logs || {
    echo "Error setting permissions on required directories"
    exit 1
}

exec "$@"
