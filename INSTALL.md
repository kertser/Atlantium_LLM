# Installation Guide

This guide provides detailed installation instructions for the Atlantium LLM system on Ubuntu server.

## System Requirements

### Hardware Requirements
- CPU: 4+ cores recommended (minimum 2 cores)
- RAM: 16GB recommended (8GB minimum)
- Storage: 20GB+ available space (SSD recommended)
- Network: Stable internet connection
- GPU (Optional): NVIDIA GPU with 8GB+ VRAM

### Software Requirements
- Ubuntu 22.04 LTS or later
- Docker 24.0+ and Docker Compose V2
- Git
- Python 3.10+ (for local development)
- curl
- OpenAI API key

## Pre-Installation Steps

### 1. System Updates
```bash
# Update package lists and upgrade existing packages
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    curl \
    git \
    python3-pip \
    software-properties-common
```

### 2. Docker Installation
```bash
# Install Docker using official script
curl -fsSL https://get.docker.com | sudo sh

# Add current user to docker group
sudo usermod -aG docker $USER

# Apply group changes (or logout/login)
newgrp docker

# Verify Docker installation
docker --version
docker compose version
```

### 3. Create Project Directory
```bash
# Create Projects directory in home folder
mkdir -p ~/Projects

# Set proper permissions
chmod 755 ~/Projects
```

## Installation Methods

### Method 1: CPU-Only Installation

1. Clone and configure:
```bash
# Clone repository
cd ~/Projects
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Set proper permissions
chmod 755 .

# Configure environment
cp .env.example .env
echo "USE_CPU=1" >> .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" >> .env
```

2. Deploy:
```bash
# Make scripts executable
chmod +x deploy.sh install_requirements.sh scripts/update_rag.sh

# Deploy application
./deploy.sh --init
```

### Method 2: GPU-Enabled Installation

1. Install NVIDIA Drivers:
```bash
# Check currently installed drivers
nvidia-smi || echo "No NVIDIA drivers detected"

# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Reboot system
sudo reboot
```

2. Install NVIDIA Container Toolkit:
```bash
# Add NVIDIA repository key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify NVIDIA Docker installation
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

3. Deploy Application:
```bash
# Clone repository
cd ~/Projects
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Set proper permissions
chmod 755 .

# Configure environment
cp .env.example .env
echo "OPENAI_API_KEY=your_api_key_here" >> .env

# Make scripts executable
chmod +x deploy.sh install_requirements.sh scripts/update_rag.sh

# Deploy application
./deploy.sh --init
```

## Post-Installation Setup

### 1. Directory Structure
After installation, verify the following directory structure:
```
~/Projects/Atlantium_LLM/
├── RAG_Data/              # Created during initialization
│   ├── stored_images/
│   ├── stored_text_chunks/
│   ├── faiss_index.bin
│   └── metadata.json
├── Raw Documents/         # For document storage
└── logs/                  # System logs
```

### 2. Permissions
```bash
# Set correct ownership
sudo chown -R $USER:$USER ~/Projects/Atlantium_LLM

# Set correct permissions
find ~/Projects/Atlantium_LLM -type d -exec chmod 755 {} \;
find ~/Projects/Atlantium_LLM -type f -exec chmod 644 {} \;
chmod +x ~/Projects/Atlantium_LLM/*.sh
chmod +x ~/Projects/Atlantium_LLM/scripts/*.sh
```

### 3. Environment Configuration
Edit `.env` file with necessary values:
```bash
nano ~/Projects/Atlantium_LLM/.env
```

Required settings:
```plaintext
OPENAI_API_KEY=your_api_key_here
CONTAINER_NAME=atlantium_llm-web-app-1
USE_CPU=0  # Set to 1 for CPU-only mode

# Optional: For GitHub webhook integration
GITHUB_WEBHOOK_SECRET=your_webhook_secret
```

## Verification

### 1. Check Service Status
```bash
# Verify container is running
docker ps | grep atlantium_llm-web-app

# Check container logs
docker logs -f atlantium_llm-web-app-1 --tail=100
```

### 2. Test Web Interface
```bash
# Test local access
curl http://localhost:9000

# For remote access, replace localhost with server IP
curl http://server_ip:9000
```

### 3. Verify GPU Support (if applicable)
```bash
# Check GPU visibility in container
docker exec atlantium_llm-web-app-1 nvidia-smi
```

## Troubleshooting

### Common Issues

1. Permission Problems
```bash
# Fix docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix directory permissions
sudo chown -R $USER:$USER ~/Projects/Atlantium_LLM
chmod -R 755 ~/Projects/Atlantium_LLM
```

2. GPU Issues
```bash
# Check NVIDIA driver status
nvidia-smi

# Check NVIDIA Docker runtime
sudo docker info | grep nvidia
```

3. Memory Issues
```bash
# Monitor memory usage
free -h
docker stats

# Adjust batch size in config.py if needed:
BATCH_SIZE: int = 2  # Lower value for less memory usage
```

4. Container Startup Issues
```bash
# Check container logs
docker logs atlantium_llm-web-app-1

# Remove and recreate containers
docker compose down -v
./deploy.sh
```

## Maintenance

### Updates
```bash
cd ~/Projects/Atlantium_LLM

# Pull latest changes
git pull

# Rebuild and restart
./deploy.sh
```

### Backup
```bash
# Create backup directory
mkdir -p ~/backups/atlantium_llm

# Backup data and configurations
cd ~/Projects/Atlantium_LLM
tar -czf ~/backups/atlantium_llm/backup_$(date +%Y%m%d).tar.gz \
    RAG_Data/ \
    "Raw Documents/" \
    logs/ \
    .env
```

### System Cleanup
```bash
# Stop containers
docker compose down

# Remove volumes
docker compose down -v

# Full system cleanup
docker system prune -af
```

### Log Management
```bash
# View application logs
tail -f ~/Projects/Atlantium_LLM/logs/system.log

# View container logs
docker logs -f atlantium_llm-web-app-1 --tail=100

# Clean old logs
find ~/Projects/Atlantium_LLM/logs -name "*.log.*" -mtime +30 -delete
```

## Support

For technical support or issues not covered in this guide, contact [Mike Kertser](mailto:mikek@atlantium.com).
