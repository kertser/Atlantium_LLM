# Atlantium AI RAG-Based Technical Assistant for UV Systems

Atlantium's Retrieval-Augmented Generation (RAG) system is an advanced AI-driven technical assistant designed for UV system management. This platform combines FAISS indexing, CLIP embeddings, and Large Language Models to process technical documentation and images, providing intelligent responses to queries about UV systems.

## Core Features

- Document Processing: PDF, DOCX, and XLSX support
- Image Analysis: Technical image classification and context extraction
- Vector Search: FAISS-powered similarity matching
- Multimodal Understanding: CLIP-based embeddings
- Context-Aware Responses: LLM with RAG enhancement
- Web Interface: Real-time interaction and document management

## Quick Installation

### Prerequisites
- Ubuntu 22.04 LTS or later
- Docker 24.0+ and Docker Compose V2
- 16GB RAM (8GB minimum)
- 20GB available storage
- OpenAI API key

### Basic Setup
```bash
# Create project directory
mkdir -p ~/Projects && cd ~/Projects

# Clone repository
git clone https://github.com/kertser/Atlantium_LLM.git
cd Atlantium_LLM

# Configure environment
cp .env.example .env
# Add your OpenAI API key to .env

# Set permissions
sudo chmod +x deploy.sh

# Deploy application
sudo ./deploy.sh --init
(or sudo ./deploy.sh for subsequent runs)
```
# Access local web interface

The web interface will be available at `http://localhost:9000`

## Project Basic Structure

```
Atlantium_LLM/
├── config.py                 # System configuration
├── server.py                 # FastAPI server
├── RAG_processor.py          # Document processor
├── run.py                    # Server runner
├── models/                   # AI components
├── utils/                    # Utility modules
├── static/                   # Frontend assets
└── docs/                     # Documentation
```
For full details, see the [Technical Reference](docs/technical-reference.md#project-full-structure)

## Key Components

### Server
- FastAPI-based REST API
- Real-time query processing
- Document management
- Chat interface with history

### RAG System
- Document ingestion and processing
- Vector indexing with FAISS
- Multimodal embeddings using CLIP
- Context-aware response generation

### Document Processing
- Text extraction from multiple formats
- Image extraction and analysis
- Technical content classification
- Context preservation

### API Endpoints

#### Document Management
- POST `/upload/document`: Upload new document
- POST `/process/documents`: Process uploaded files
- GET `/get/documents`: List available documents
- DELETE `/delete/document`: Remove document

#### Query Processing
- POST `/query/text`: Process text queries
- POST `/query/image`: Process image queries
- POST `/chat/reset`: Reset chat history
- GET `/chat/history`: Retrieve chat history

## Documentation

- [Installation Guide](docs/installation.md)
- [Technical Reference](docs/technical-reference.md)
- [Frontend Guide](docs/frontend.md)
- [Models Documentation](docs/models.md)
- [Utils Reference](docs/utils.md)
- [Update Service](docs/update-service.md)

## Support

For technical assistance, contact [Mike Kertser](mailto:mikek@atlantium.com)