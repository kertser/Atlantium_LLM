# Atlantium AI RAG-Based Technical Assistant for UV Systems

Atlantium's Retrieval-Augmented Generation (RAG) system is an advanced AI-driven technical assistant designed for UV system management. This platform combines FAISS indexing, CLIP embeddings, and GPT-based analysis to process text and images for efficient query handling, document management, and image analysis.

## Features

### Core Capabilities
- **Text and Image Processing**:
  - Extracts and analyzes text and images from PDF, DOCX, and XLSX files
  - Uses FAISS for vector-based similarity searches
  - Implements CLIP for multimodal embeddings
- **AI-Powered Responses**:
  - Integrates with OpenAI and Anthropic APIs
  - Provides context-aware technical responses
  - Handles both text and image queries
- **Document Management**:
  - Web interface for document uploads and organization
  - Automatic content indexing and retrieval
  - Folder-based document organization
- **Image Analysis**:
  - Technical diagram recognition
  - Image deduplication using perceptual hashing
  - Context-aware image retrieval

## Quick Start

### CPU-Only Installation
```bash
# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure for CPU usage
cp .env.example .env
echo "USE_CPU=1" >> .env

# Deploy
chmod +x deploy.sh
./deploy.sh --init
```

### GPU-Enabled Installation
```bash
# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure
cp .env.example .env

# Deploy
chmod +x deploy.sh
./deploy.sh --init
```

For detailed installation instructions, requirements, and troubleshooting, see [INSTALL.md](INSTALL.md).

## System Requirements

### Minimum Requirements
- Docker 24.0+ and Docker Compose V2
- 16GB RAM recommended (8GB minimum)
- Ubuntu 22.04+ or compatible Linux distribution
- OpenAI API key
- 20GB available disk space

### GPU Support (Optional but Recommended)
- NVIDIA GPU with CUDA 12.1+ support
- 8GB+ GPU memory
- NVIDIA Container Toolkit
- NVIDIA drivers installed and configured

## Basic Usage

### Web Interface
Access the web interface at `http://server_ip:9000`

### Document Management
```bash
# Upload document via CLI
curl -X POST "http://server_ip:9000/upload/document" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/document.pdf" \
    -F "folder=your/folder/path"

# Trigger processing
curl -X POST "http://server_ip:9000/process/documents"
```

## Basic Maintenance

### Service Commands
```bash
# View logs
docker logs atlantium_llm-web-app-1 -f --tail=100

# Restart service
docker compose restart

# Update deployment
git pull
./deploy.sh

# Full cleanup
docker system prune --all --volumes --force
```

### Quick Troubleshooting

1. **GPU Issues**: Verify with `nvidia-smi`
2. **Memory Issues**: Check with `free -h`
3. **Permission Issues**: Run `sudo chown -R $USER:$USER .`

See [INSTALL.md](INSTALL.md) for comprehensive troubleshooting.

## Automated Updates

1. Configure GitHub webhook:
   - URL: `http://server_ip:9000/webhook`
   - Content type: `application/json`

2. Add to `.env`:
```plaintext
GITHUB_WEBHOOK_SECRET=your_webhook_secret
```

## Support

For technical support, contact [Atlantium Technologies Support](mailto:support@atlantium.com)

## License

Private commercial software. All rights reserved to Atlantium Technologies Ltd.
