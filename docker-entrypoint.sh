#!/bin/bash
set -e

# Function to wait for a service - not used in single service, but remains here for future use
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local timeout=30

    echo "Waiting for $service to be ready..."
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port"; then
            echo "$service is ready!"
            return 0
        fi
        echo "Waiting for $service... $i/$timeout"
        sleep 1
    done
    echo "$service is not available"
    return 1
}

# Initialize RAG database if requested
if [ "$INITIALIZE_RAG" = "true" ]; then
    echo "Initializing RAG database..."
    python -c "from utils.initialize_RAG import initialize_rag_database; initialize_rag_database()"
fi

# Create required directories if they don't exist
mkdir -p "RAG_Data/stored_images" "Raw Documents" logs

# Set proper permissions
chmod -R 755 "RAG_Data" "Raw Documents" logs

# Execute the main command
exec "$@"