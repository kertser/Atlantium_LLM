#!/bin/bash
set -e  # Exit on errors

# Only initialize if INITIALIZE_RAG environment variable is set to true
if [ "$INITIALIZE_RAG" = "true" ]; then
    echo "Initializing RAG database..."
    python -c "from utils.initialize_RAG import initialize_rag_database; initialize_rag_database()"
else
    echo "Skipping RAG database initialization..."
fi

# Execute the main command
exec "$@"