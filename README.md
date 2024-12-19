# Atlantium AI RAG-Based Technical Assistant for UV Systems

Atlantium's Retrieval-Augmented Generation (RAG) system is an advanced AI-driven technical assistant designed for UV system management. This platform combines FAISS indexing, CLIP embeddings, and Large Language Models to process technical documentation and images, providing intelligent responses to queries about UV systems.

## Features

### Core Capabilities
- **Technical Documentation Processing**:
  - Extracts and analyzes text and images from PDF, DOCX, and XLSX files
  - Maintains document context and relationships
  - Supports hierarchical document organization
  
- **Advanced Search and Retrieval**:
  - FAISS-powered vector similarity search
  - CLIP-based multimodal embeddings
  - Context-aware query processing
  
- **AI-Powered Responses**:
  - Integration with OpenAI GPT models
  - Technical context preservation
  - Multi-turn conversation support
  
- **Intelligent Image Analysis**:
  - Technical diagram and schematic recognition
  - Automatic relevance assessment
  - Perceptual deduplication
  
- **Document Management System**:
  - Web-based interface for document organization
  - Folder hierarchy support
  - Batch processing capabilities

## Deployment Options

### Prerequisites
- Ubuntu 22.04 LTS or later
- Docker 24.0+ and Docker Compose V2
- 16GB RAM (8GB minimum)
- 20GB available storage
- OpenAI API key
- Git

### Basic Installation
```bash
# Create project directory
mkdir -p ~/Projects && cd ~/Projects

# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Set up environment
cp .env.example .env
nano .env  # Add your OpenAI API key and other settings

# Make scripts executable
chmod +x deploy.sh install_requirements.sh

# Deploy application
./deploy.sh --init
```

### GPU Support (Optional)
Additional requirements for GPU acceleration:
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Driver 525 or later
- NVIDIA Container Toolkit

Enable GPU support in `.env`:
```plaintext
USE_CPU=0  # Set to 1 for CPU-only mode
```

## Usage Guide

### Web Interface
The web interface is available at `http://your_server:9000` after deployment.
(local deployment - `http://localhost:9000`)

### Document Upload
1. **Via Web Interface**:
   - Navigate to `http://your_server:9000`
   - Use the upload interface to add documents
   - Click "Process Documents" to index new content

2. **Via Command Line**:
```bash
# Upload document
curl -X POST "http://your_ip:9000/upload/document" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/document.pdf" \
    -F "folder=your/folder/path"

# Process uploaded documents
curl -X POST "http://your_server:9000/process/documents"
```

## Maintenance

### Common Commands
```bash
# View application logs
docker logs atlantium_llm-web-app-1 -f --tail=100

# Restart service
docker compose restart

# Update deployment
git pull
./deploy.sh

# System cleanup
docker system prune --all --volumes --force
```

### Automated Updates
Configure GitHub webhook for automatic updates:

1. In GitHub repository settings:
   - Webhook URL: `http://your_server:9000/webhook`
   - Content type: `application/json`
   - Secret: Create a secure webhook secret

2. Add to `.env`:
```plaintext
GITHUB_WEBHOOK_SECRET=your_webhook_secret
```

## Troubleshooting

### Common Issues
1. **Permission Problems**:
```bash
sudo chown -R $USER:$USER ~/Projects/Atlantium_LLM
chmod -R 755 ~/Projects/Atlantium_LLM
```

2. **GPU Detection Issues**:
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check container toolkit
nvidia-container-cli info
```

3. **Memory Issues**:
```bash
# Check available memory
free -h

# Monitor container resources
docker stats atlantium_llm-web-app-1
```

For detailed installation instructions and advanced configuration options, see [INSTALL.md](INSTALL.md).

For technical architecture and implementation details, see [technical-reference.md](docs/technical-reference.md).

## Support

For technical assistance, contact [Mike Kertser](mailto:mikek@atlantium.com).

## License

This software is proprietary and confidential. All rights reserved to Atlantium Technologies Ltd.

---
For detailed documentation on deployment, configuration, and troubleshooting, please refer to the documentation files in the `docs` directory.