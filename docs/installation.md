# Installation Guide

This guide provides detailed installation instructions for the Atlantium LLM system. For a quick overview, see our [main README](../README.md).

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
- OpenAI API key

For technical details about system components, see our [Technical Reference](technical-reference.md).

## Installation Methods

### Method 1: Standard Installation

1. **System Updates**:
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

2. **Docker Setup**:
```bash
# Install Docker using official script
curl -fsSL https://get.docker.com | sudo sh

# Add current user to docker group
sudo usermod -aG docker $USER

# Apply group changes
newgrp docker

# Verify installation
docker --version
docker compose version
```

3. **Application Installation**:
```bash
# Create Projects directory
mkdir -p ~/Projects
cd ~/Projects

# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure environment
cp .env.example .env
echo "OPENAI_API_KEY=your_api_key_here" >> .env

# Deploy
./deploy.sh --init
```

### Method 2: GPU-Enabled Installation

1. **NVIDIA Driver Installation**:
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

2. **NVIDIA Container Toolkit**:
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
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

3. **Follow Standard Installation** steps above, but ensure `USE_CPU=0` in your `.env` file.

## Post-Installation Setup

### 1. Directory Structure
```
~/Projects/Atlantium_LLM/
├── docs/                 # Documentation
├── scripts/             # System scripts
│   └── update_service/  # Update service components
├── RAG_Data/            # Generated during initialization
├── Raw Documents/       # Document storage
└── logs/               # System logs
```

### 2. Environment Configuration
Configure your `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
CONTAINER_NAME=atlantium_llm-web-app-1
USE_CPU=0  # Set to 1 for CPU-only mode
```

### 3. Update Service Setup
For automatic updates setup, see our [Update Service Guide](update-service.md).

## Verification

### 1. System Check
```bash
# Check container status
docker ps | grep atlantium_llm-web-app

# View logs
docker logs -f atlantium_llm-web-app-1
```

### 2. Web Interface
Access the web interface at `http://localhost:9000`

For web interface details, see our [Frontend Documentation](frontend.md).

### 3. GPU Verification (if applicable)
```bash
# Check GPU status
docker exec atlantium_llm-web-app-1 nvidia-smi
```

## Troubleshooting

### Common Issues

1. **Docker Permission Issues**:
```bash
# Fix permissions
sudo usermod -aG docker $USER
newgrp docker
```

2. **Memory Problems**:
```bash
# Check memory usage
free -h
docker stats
```

3. **GPU Issues**:
```bash
# Verify NVIDIA setup
nvidia-smi
docker info | grep nvidia
```

For additional technical details, see our [Technical Reference](technical-reference.md).

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

### Updates
For system updates, refer to our [Update Service Guide](update-service.md).

### Log Management
```bash
# View logs
tail -f ~/Projects/Atlantium_LLM/logs/system.log

# Clean old logs
find ~/Projects/Atlantium_LLM/logs -name "*.log.*" -mtime +30 -delete
```

## Next Steps

- Set up the [Update Service](update-service.md)
- Explore the [Frontend Interface](frontend.md)
- Review [Technical Documentation](technical-reference.md)
- Learn about [AI Models](models.md)

## Support

For technical support, contact [Mike Kertser](mailto:mikek@atlantium.com).