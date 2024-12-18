#!/bin/bash
set -e

# Check if GPU requirements exist and are valid
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

echo "Installing from: $REQ_FILE"
pip install -r "$REQ_FILE" || {
    echo "Error installing dependencies. Check for version compatibility issues."
    exit 1
}

echo "Dependencies installed successfully"
