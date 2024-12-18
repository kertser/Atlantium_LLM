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

# Ensure compatibility with Python versions
if [[ "$USE_CPU" == "0" && "$(python3 --version | grep -Eo '[0-9]+\.[0-9]+')" == "3.12" ]]; then
    echo "Warning: faiss-gpu is not available for Python 3.12. Skipping faiss-gpu installation."
    sed -i '/faiss-gpu/d' requirements_gpu.txt
fi

# Enable BuildKit for Docker builds
export DOCKER_BUILDKIT=1

# Build and start containers
if ! docker-compose build --no-cache; then
    echo "Build failed. Retrying with CPU configuration."
    export BUILD_TYPE=cpu
    docker-compose build --no-cache
fi

if docker-compose up -d; then
    echo "Deployment complete. Service available at http://localhost:9000"
else
    echo "Error: Deployment failed."
    exit 1
fi