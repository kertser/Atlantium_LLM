# Installation Guide

This document provides detailed installation instructions for the Atlantium LLM system.

## System Requirements

### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 16GB recommended (8GB minimum)
- Storage: 20GB+ available space
- GPU (Optional): NVIDIA GPU with 8GB+ VRAM

### Software Requirements
- Ubuntu 22.04+ or compatible Linux distribution
- Docker 24.0+
- Docker Compose V2
- Git
- Python 3.10+ (for local development)

## Installation Methods

Choose one of the following installation methods:

### Method 1: CPU-Only Installation

1. Setup basic requirements:
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

2. Deploy the application:
```bash
# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure environment
cp .env.example .env
echo "USE_CPU=1" >> .env

# Deploy
chmod +x deploy.sh
./deploy.sh --init
```

### Method 2: GPU-Enabled Installation

1. Install NVIDIA Drivers:
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

2. Install NVIDIA Container Toolkit:
```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

3. Deploy the application:
```bash
# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure environment
cp .env.example .env

# Deploy
chmod +x deploy.sh
./deploy.sh --init
```

## Troubleshooting

### Common Issues

1. GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# If drivers are missing, install them:
sudo ubuntu-drivers autoinstall
sudo reboot
```

2. Container Toolkit Issues
```bash
# Reinstall toolkit
sudo apt-get remove nvidia-container-toolkit
sudo apt-get install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

4. Memory Issues
```bash
# Check system memory
free -h

# If needed, adjust batch size in config.py:
BATCH_SIZE: int = 2  # Reduce from default
```

### Verifying Installation

1. Check service status:
```bash
docker ps
```

2. Check logs:
```bash
docker logs atlantium_llm-web-app-1
```

3. Test web interface:
```bash
curl http://localhost:9000
```

## Environment Configuration

### Required Environment Variables

Edit `.env` file with appropriate values:
```bash
# API Keys
OPENAI_API_KEY=your_api_key_here

# Deployment Settings
CONTAINER_NAME=atlantium_llm-web-app-1
USE_CPU=0  # Set to 1 for CPU-only mode

# Optional: GitHub Webhook
GITHUB_WEBHOOK_SECRET=your_webhook_secret
```

### Optional Configuration

Adjust `config.py` for fine-tuning:
```python
BATCH_SIZE: int = 5  # Adjust based on available memory
USE_GPU: bool = True  # GPU usage flag
SERVER_PORT: int = 9000  # Service port
```

## Maintenance

### Updates
```bash
# Pull latest changes
git pull

# Rebuild and restart
./deploy.sh
```

### Backup
```bash
# Backup data directories
tar -czf backup_$(date +%Y%m%d).tar.gz RAG_Data/ "Raw Documents/"
```

### Cleanup
```bash
# Remove all containers and volumes
docker-compose down -v

# Clean Docker system
docker system prune -af
```
