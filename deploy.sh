#!/bin/bash

## First-time deployment:
# chmod +x deploy.sh
# ./deploy.sh --init

## Regular deployment:
# ./deploy.sh

# Check for GPU availability
check_gpu() {
    if [ "$(uname)" = "Linux" ]; then
        if lspci | grep -E "VGA|3D|Display" > /dev/null; then
            return 0  # GPU available
        fi
    fi
    return 1  # No GPU available
}

# Set build type based on GPU availability
if check_gpu; then
    echo "GPU detected - using GPU configuration"
    export BUILD_TYPE=gpu
    export USE_CPU=0
else
    echo "No GPU detected - using CPU configuration"
    export BUILD_TYPE=cpu
    export USE_CPU=1
fi

# First time setup check
if [ "$1" == "--init" ]; then
    echo "Performing first-time initialization..."
    export INITIALIZE_RAG=true
fi

# Build and start containers
docker-compose build --no-cache
docker-compose up -d

echo "Deployment complete. Service available at http://localhost:9000"