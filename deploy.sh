#!/bin/bash

## First-time deployment:
# chmod +x deploy.sh
# ./deploy.sh --init

## Regular deployment:
# ./deploy.sh

# Check for GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q .; then
            return 0  # GPU available
        fi
    fi
    return 1  # No GPU available
}

# Set build type based on GPU availability
if check_gpu; then
    echo "GPU detected - using GPU configuration"
    export BUILD_TYPE=gpu
    
    # Check for nvidia-docker
    if ! command -v nvidia-docker &> /dev/null && ! docker info | grep -q "Runtimes:.*nvidia"; then
        echo "Warning: NVIDIA Docker runtime not found. Please install nvidia-docker2"
        echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
else
    echo "No GPU detected - using CPU configuration"
    export BUILD_TYPE=cpu
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