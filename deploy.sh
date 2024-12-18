#!/bin/bash

## First-time deployment:
# chmod +x deploy.sh
# ./deploy.sh --init

## Regular deployment:
# ./deploy.sh

# Function to check NVIDIA driver installation
check_nvidia_drivers() {
    if [ -f "/proc/driver/nvidia/version" ]; then
        if nvidia-smi &> /dev/null; then
            return 0  # Drivers are properly installed
        fi
    fi
    return 1  # Drivers are not installed
}

# Function to check CUDA availability
check_cuda() {
    if command -v nvcc &> /dev/null; then
        return 0  # CUDA is available
    fi
    return 1  # CUDA is not available
}

# Function to check container toolkit
check_container_toolkit() {
    if command -v nvidia-container-cli &> /dev/null; then
        # Test with a basic CUDA container
        if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            return 0  # Container toolkit is working
        fi
    fi
    return 1  # Container toolkit is not working
}

# Enhanced GPU detection
detect_gpu_configuration() {
    echo "Checking GPU configuration..."

    # Check for NVIDIA drivers
    if ! check_nvidia_drivers; then
        echo "Warning: NVIDIA drivers not found or not properly installed."
        echo "Falling back to CPU configuration."
        return 1
    fi

    # Check for NVIDIA Container Toolkit
    if ! check_container_toolkit; then
        echo "Warning: NVIDIA Container Toolkit not found or not properly configured."
        echo "Falling back to CPU configuration."
        return 1
    fi

    # If we get here, GPU configuration is valid
    echo "GPU configuration verified successfully."
    return 0
}

# Main deployment logic
main() {
    # Check if initialization is requested
    INITIALIZE=""
    if [ "$1" = "--init" ]; then
        INITIALIZE="true"
        echo "Initialization requested"
    fi

    # Detect GPU and set configuration
    if detect_gpu_configuration; then
        echo "GPU detected - using GPU configuration"
        export BUILD_TYPE=gpu
        export USE_CPU=0
    else
        echo "Using CPU configuration"
        export BUILD_TYPE=cpu
        export USE_CPU=1
    fi

    # Ensure compatibility with Python versions
    if [[ "$USE_CPU" == "0" && "$(python3 --version | grep -Eo '[0-9]+\.[0-9]+')" == "3.12" ]]; then
        echo "Warning: faiss-gpu is not available for Python 3.12. Skipping faiss-gpu installation."
        sed -i '/faiss-gpu/d' requirements_gpu.txt
    fi

    # Enable BuildKit, fallback if Buildx is missing
    if ! docker buildx version > /dev/null 2>&1; then
        echo "Buildx not found or broken. Disabling BuildKit."
        export DOCKER_BUILDKIT=0
    else
        export DOCKER_BUILDKIT=1
    fi

    # Build containers
    echo "Building containers..."
    if ! docker-compose build --no-cache; then
        echo "Build failed. Retrying with CPU configuration."
        export BUILD_TYPE=cpu
        export USE_CPU=1
        if ! docker-compose build --no-cache; then
            echo "Error: Build failed even with CPU configuration."
            exit 1
        fi
    fi

    # Set initialization environment variable if needed
    if [ -n "$INITIALIZE" ]; then
        export INITIALIZE_RAG=true
    fi

    # Start containers
    echo "Starting containers..."
    if docker-compose up -d; then
        echo "Deployment complete. Service available at http://localhost:9000"

        # Wait for service to be ready
        echo "Waiting for service to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:9000 > /dev/null; then
                echo "Service is ready!"
                break
            fi
            sleep 2
        done
    else
        echo "Error: Deployment failed."
        exit 1
    fi
}

# Execute main with all arguments
main "$@"