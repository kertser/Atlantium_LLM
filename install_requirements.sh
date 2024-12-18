#!/bin/bash
set -e

check_gpu() {
    if [ "$(uname)" = "Linux" ]; then
        # Look for GPU-related PCI devices:
        if lspci | grep -E "VGA|3D|Display" > /dev/null; then
            return 0
        fi
    fi
    return 1
}

if [ -n "$USE_CPU" ]; then
    if [ "$USE_CPU" = "1" ]; then
        echo "CPU mode explicitly set - installing CPU dependencies"
        REQ_FILE="requirements_cpu.txt"
    else
        echo "GPU mode explicitly set - installing GPU dependencies"
        REQ_FILE="requirements_gpu.txt"
    fi
else
    if check_gpu; then
        echo "GPU detected (via lspci) - installing GPU dependencies"
        export USE_CPU=0
        REQ_FILE="requirements_gpu.txt"
    else
        echo "No GPU detected or non-Linux system - installing CPU dependencies"
        export USE_CPU=1
        REQ_FILE="requirements_cpu.txt"
    fi
fi

# Safety check if the requirements file exists
if [ ! -f "$REQ_FILE" ]; then
    echo "Error: $REQ_FILE not found!"
    exit 1
fi

echo "Installing from: $REQ_FILE"
echo "USE_CPU=$USE_CPU"
pip install -r "$REQ_FILE"

echo "Dependencies installed successfully"
