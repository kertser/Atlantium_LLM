# Atlantium RAG Update System

## Overview
This system provides automated updates for the Atlantium RAG application through a systemd service. It monitors the GitHub release branch for new tagged versions and automatically updates the Docker-based installation while preserving all data volumes.

## For Developers

### Branch Structure
```
main
  ↑
release
  ↑
development
  ↑
feature branches
```

- `development`: Active development branch - all new features start here
- `release`: Staging and testing branch
- `main`: Production code (managed through releases)

### Development Workflow

1. **Start New Development**:
```bash
# Ensure you're up to date
git checkout development
git pull origin development

# Create new feature branch
git checkout -b feature/your-feature
```

2. **Work on Your Feature**:
```bash
# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/your-feature

# Create Pull Request on GitHub
# development <- feature/your-feature
```

3. **Create a Release**:
```bash
# Update development first
git checkout development
git pull origin development

# Switch to release branch
git checkout release
git pull origin release

# Merge development into release
git merge development

# Create and push tag
git tag v1.0.1
git push origin release
git push origin v1.0.1
```

### Version Tagging
Use semantic versioning: `vMAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible

Example: `v1.2.3`

## For Users

### Prerequisites
- Ubuntu 22.04+ or compatible Linux distribution
- Docker 24.0+ and Docker Compose V2
- Git
- Systemd
- NVIDIA drivers and container toolkit (for GPU support)

### Initial Setup
1. **Install Docker** (if not already installed):
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

2. **Install NVIDIA Container Toolkit** (for GPU support):
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Quick Installation
```bash
# Clone repository
cd ~/Projects
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Run installer
sudo ./install.sh
```

### Custom Installation
If you want to install to a different location:
```bash
sudo APP_DIR=/your/custom/path ./install.sh
```

### Directory Structure
```
~/Projects/Atlantium_LLM/
├── logs/
│   └── updates/        # Update logs
├── backups/           # Code backups (not data)
├── scripts/           # Update scripts
├── RAG_Data/         # Persisted RAG data (Docker volume)
├── Raw Documents/    # Persisted documents (Docker volume)
└── ... (application files)
```

### Managing the Service
```bash
# View service status
systemctl status atlantium-update

# View update logs
tail -f ~/Projects/Atlantium_LLM/logs/updates/update.log

# View container logs
docker logs -f atlantium_llm-web-app-1

# Manual update check
sudo systemctl restart atlantium-update
```

### Troubleshooting

1. **Docker Permission Issues**:
```bash
# Add yourself to docker group
sudo usermod -aG docker $USER
newgrp docker  # Or log out and back in
```

2. **GPU Detection Issues**:
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify container toolkit
nvidia-container-cli info
```

3. **Volume Persistence**:
```bash
# List Docker volumes
docker volume ls

# Check volume data
docker volume inspect raw_docs
docker volume inspect rag_data
```

4. **Service Issues**:
```bash
# Check service logs
journalctl -u atlantium-update -f

# Check service configuration
systemctl cat atlantium-update
```

### Uninstallation
```bash
# Stop and remove service
sudo systemctl stop atlantium-update
sudo systemctl disable atlantium-update
sudo rm /etc/systemd/system/atlantium-update.service
sudo systemctl daemon-reload

# Optional: Remove Docker volumes (CAUTION: removes all data)
docker-compose down -v
```

## Data Persistence

The update service preserves:
- All RAG data (embeddings, indexes)
- Uploaded documents
- System logs
- Configuration files

Docker volumes are not affected during updates.

## Support

If you encounter issues:
1. Check all logs (service, Docker, application)
2. Verify Docker and GPU configuration
3. Ensure volumes are properly mounted
4. Create a GitHub issue with:
   - Full error logs
   - System information
   - Docker and NVIDIA information