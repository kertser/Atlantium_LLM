# Atlantium AI RAG-Based Technical Assistant for UV Systems

Atlantium's Retrieval-Augmented Generation (RAG) system is an advanced AI-driven technical assistant designed for UV system management. This platform combines FAISS indexing, CLIP embeddings, and Large Language Models to process technical documentation and images, providing intelligent responses to queries about UV systems.

## Quick Start

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

# Deploy application
./deploy.sh --init
```

For detailed installation instructions, see our [Installation Guide](docs/installation.md).

## Key Features

### Technical Documentation Processing
- Extracts and analyzes text and images from PDF, DOCX, and XLSX files
- Maintains document context and relationships
- Supports hierarchical document organization

See [Technical Reference](docs/technical-reference.md) for implementation details.

### AI-Powered Search and Analysis
- FAISS-powered vector similarity search
- CLIP-based multimodal embeddings
- Context-aware query processing

Learn more about our [AI Models](docs/models.md).

### Web Interface
- Document management system
- Interactive chat interface
- Image analysis capabilities

Explore our [Frontend Documentation](docs/frontend.md).

### Automatic Updates
- Git-based version control
- Automated deployment
- Data persistence

Check out our [Update Service Guide](docs/update-service.md).

## Documentation

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Technical Reference](docs/technical-reference.md) - System architecture and components
- [Frontend Guide](docs/frontend.md) - Web interface implementation
- [Models Documentation](docs/models.md) - AI models and prompts
- [Utils Reference](docs/utils.md) - Utility functions
- [Update Service](docs/update-service.md) - Automatic update system

## Support

For technical assistance, contact [Mike Kertser](mailto:mikek@atlantium.com).