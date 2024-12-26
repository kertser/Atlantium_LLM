# Installation Guide

## System Requirements

### Hardware Requirements
- CPU: 4+ cores recommended (minimum 2 cores)
- RAM: 16GB recommended (8GB minimum)
- Storage: 20GB+ available space (SSD recommended)
- GPU: NVIDIA GPU with 8GB+ VRAM (optional)
- Network: Stable internet connection

### Software Requirements
- Ubuntu 22.04 LTS or later
- Docker 24.0+ and Docker Compose V2
- Git
- Python 3.10+ (for local development)
- OpenAI API key

## Installation Methods

### Standard Installation

1. System Updates
```bash
# Update package lists
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    curl \
    git \
    python3-pip \
    software-properties-common
```

2. Docker Setup
```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

3. Application Setup
```bash
# Create project directory
mkdir -p ~/Projects && cd ~/Projects

# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure environment
cp .env.example .env
# Add OpenAI API key to .env

# Set permissions
sudo chmod +x deploy.sh

# Deploy
sudo ./deploy.sh --init
```

### GPU-Enabled Installation

1. NVIDIA Driver Installation
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

2. NVIDIA Container Toolkit
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Configuration

### Environment Setup
Configure your `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
CONTAINER_NAME=atlantium_llm-web-app-1
USE_CPU=0  # Set to 1 for CPU-only mode
```

### Directory Structure
```plaintext
~/Projects/Atlantium_LLM/
├── RAG_Data/            # Generated during initialization
├── Raw Documents/       # Document storage
├── logs/                # System logs
└── scripts/             # System scripts
```

## Verification

### System Check
```bash
# Check container status
docker ps | grep atlantium_llm-web-app

# View logs
docker logs -f atlantium_llm-web-app-1

# Access web interface
http://localhost:9000
```

### GPU Verification
```bash
# Check GPU status
nvidia-smi
docker exec atlantium_llm-web-app-1 nvidia-smi
```

## Common Issues

### Docker Permission Issues
```bash
# Fix permissions
sudo usermod -aG docker $USER
newgrp docker
```

### Memory Problems
```bash
# Check memory usage
free -h
docker stats
```

### GPU Issues
```bash
# Verify NVIDIA setup
nvidia-smi
nvidia-container-toolkit-cli info
```

### Container Issues
```bash
# Remove containers and volumes
docker-compose down -v

# Clean Docker system
docker system prune --all --volumes --force

# Rebuild containers
sudo ./deploy.sh --init
```

## Maintenance

### Backups
```bash
# Create backup
cd ~/Projects/Atlantium_LLM
tar -czf ~/backups/atlantium_backup_$(date +%Y%m%d).tar.gz \
    RAG_Data/ \
    "Raw Documents/" \
    .env
```

### Log Management
```bash
# View logs
tail -f ~/Projects/Atlantium_LLM/logs/system.log

# Clean old logs
find ~/Projects/Atlantium_LLM/logs -name "*.log.*" -mtime +30 -delete
```

### Updates
For system updates and maintenance procedures, refer to the [Update Service Guide](../docs/update-service.md).

## Related Documentation

- [Technical Reference](../docs/technical-reference.md)
- [Frontend Documentation](../docs/frontend.md)
- [Models Documentation](../docs/models.md)
- [Utils Documentation](../docs/utils.md)