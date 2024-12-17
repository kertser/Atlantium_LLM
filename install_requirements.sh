#!/bin/bash

# Function to check GPU availability on Linux
check_gpu() {
    if [ "$(uname)" == "Linux" ]; then
        if command -v nvidia-smi &> /dev/null; then
            return 0  # GPU available
        fi
    fi
    return 1  # No GPU available
}

# Determine if we should use CPU version
if check_gpu && [ -z "$USE_CPU" ]; then
    echo "GPU detected - installing GPU dependencies"
    export USE_CPU=0
else
    echo "No GPU detected or CPU mode forced - installing CPU dependencies"
    export USE_CPU=1
fi

# Install dependencies
pip install -r requirements.txt