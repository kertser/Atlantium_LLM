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

## System Requirements

### Minimum Requirements
- Docker and Docker Compose
- 8GB RAM (16GB recommended)
- Ubuntu Linux server
- API keys for OpenAI/Anthropic

### GPU Support (Optional)
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (for GPU deployment)
- 8GB+ GPU memory recommended

## Installation

### Prerequisites
1. Install Docker and Docker Compose
2. For GPU support, install NVIDIA Container Toolkit:
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

3. Deploy the application:
```bash
# First-time deployment with initialization
chmod +x deploy.sh
./deploy.sh --init

# Regular deployment
./deploy.sh
```

The script will automatically:
- Detect GPU availability
- Choose appropriate configuration (CPU/GPU)
- Build and start the containers
- Initialize the system (if --init flag is used)

The web interface will be available at `http://server_ip:9000`

## Document Management

### Web Interface
Upload and manage documents through the web interface at `http://server_ip:9000`

### CLI Upload
```bash
# Upload document
curl -X POST "http://server_ip:9000/upload/document" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/document.pdf" \
    -F "folder=your/folder/path"

# Trigger processing
curl -X POST "http://server_ip:9000/process/documents"
```

## Project Structure

```
Atlantium_LLM/
├── models/          # AI model implementations
├── utils/           # Utility functions and helpers
├── static/          # Web interface assets
├── RAG_Data/        # Generated indices and metadata
├── Raw Documents/   # Source documents for processing
└── logs/            # Application logs
```

See detailed documentation in the [docs](docs) directory.

## Configuration

### Environment Variables
- `BUILD_TYPE`: Set to 'gpu' or 'cpu' (auto-detected by deploy.sh)
- `INITIALIZE_RAG`: Set to 'true' for first-time setup
- `CONTAINER_NAME`: Override default container name

### Core Settings
Key settings in `config.py`:
```python
SERVER_PORT: int = 9000
RAW_DOCUMENTS_PATH: Path = Path("Raw Documents")
CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
USE_GPU: bool = True  # Automatically managed by deployment
```

## Deployment

### GitHub Webhook Setup

1. Add webhook in GitHub repository settings:
   - URL: `http://server_ip:9000/webhook`
   - Content type: `application/json`
   - Secret: Create a secure webhook secret

2. Add secret to `.env`:
```
GITHUB_WEBHOOK_SECRET=your_webhook_secret
```

### Server Management

```bash
# View logs
docker logs atlantium_llm-web-app-1 -f --tail=100

# Restart service
docker-compose restart

# Full cleanup
docker system prune --all --volumes --force

# Update deployment
./deploy.sh
```

## Development

See detailed documentation:
- [Technical Reference](docs/technical-reference.md)
- [Models Documentation](docs/models.md)
- [Utils Documentation](docs/utils.md)
- [Frontend Documentation](docs/frontend.md)

## Support

For technical support, contact [Atlantium Technologies Support](mailto:support@atlantium.com)

## License

Private commercial software. All rights reserved to Atlantium Technologies Ltd.
