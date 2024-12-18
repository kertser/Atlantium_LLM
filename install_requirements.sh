#!/bin/bash

# Handle both CPU and GPU configurations.
set -e

# Determine the requirements file based on CPU or GPU usage
REQ_FILE=""
if [[ "$USE_CPU" == "1" ]]; then
    echo "CPU mode explicitly set - installing CPU dependencies"
    REQ_FILE="requirements_cpu.txt"
else
    echo "GPU mode explicitly set - installing GPU dependencies"
    REQ_FILE="requirements_gpu.txt"
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "Error: $REQ_FILE not found!"
    exit 1
fi

# Install dependencies with the extra index URL as appropriate
if [[ "$USE_CPU" == "1" ]]; then
    echo "Installing CPU dependencies from $REQ_FILE"
    pip install --no-cache-dir -r "$REQ_FILE" --extra-index-url https://download.pytorch.org/whl/cpu || {
        echo "Error installing CPU dependencies. Check for version compatibility issues."
        exit 1
    }
else
    echo "Installing GPU dependencies from $REQ_FILE"
    pip install --no-cache-dir -r "$REQ_FILE" --extra-index-url https://download.pytorch.org/whl/cu121 || {
        echo "Error installing GPU dependencies. Check for version compatibility issues."
        exit 1
    }
fi

echo "Dependencies installed successfully"
