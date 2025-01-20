# Atlantium AI RAG-Based Technical Assistant for UV Systems

Atlantium's Retrieval-Augmented Generation (RAG) system is a cutting-edge AI platform designed for advanced UV system management. This system integrates FAISS indexing, CLIP embeddings, and Large Language Models to deliver intelligent, context-aware technical assistance for managing UV systems, documents, and images.

---

## 🚀 Key Features

### 🗂️ Document Management
- **Supported Formats**: PDF, DOCX, XLSX
- **Hierarchical File Organization**
- **Bulk Operations & Context Menus**

### 📸 Image Analysis
- **Image Classification**: Zero-shot capabilities for technical images
- **Context Extraction**: Associates images with text-based context
- **Deduplication**: Intelligent removal of duplicate images

### 🔍 Vector Search
- **Powered by FAISS**: Fast and efficient similarity matching
- **Multimodal Embeddings**: CLIP/BLIP integration for comprehensive insights

### 🤖 Context-Aware Responses
- **Integrated LLM**: Advanced language models with RAG enhancements
- **Query Support**: Processes text and image queries

### 🌐 Web Interface
- **Real-Time Interaction**: Query responses and document management
- **Chat History**: Persistent, resettable conversation tracking

---

## 📋 System Requirements

### Hardware
- **CPU**: 4+ cores recommended
- **RAM**: 16GB recommended
- **Storage**: 20GB+ (SSD preferred)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional for GPU acceleration)

### Software
- **Operating System**: Ubuntu 22.04 LTS or later
- **Tools**: Docker 24.0+ and Docker Compose V2, Git
- **API Access**: OpenAI API key

---

## 🛠️ Installation Guide

### Standard Setup

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
```

### Access Web Interface
Visit: `http://localhost:9000`

For detailed installation, refer to the [Installation Guide](docs/installation.md).

---

## 🗂️ Project Structure

```plaintext
Atlantium_LLM/
├── docs/                     # Documentation files
├── models/                   # AI models and integrations
├── utils/                    # Utility modules
├── logs/                     # Log files
├── RAG_Data/                 # RAG-generated data
├── Raw Documents/            # Document storage
├── static/                   # Frontend assets
├── scripts/                  # Management scripts
├── config.py                 # System configuration
├── deploy.sh                 # Deployment script
└── server.py                 # FastAPI server
```

For full structure details, visit the [Technical Reference](docs/technical-reference.md).

---

## 🌟 Core Modules

### 📡 Server
- **Backend**: FastAPI-based REST API
- **Capabilities**: Real-time processing, document handling, chat interface

### 📚 RAG System
- **Document Ingestion**: Processes PDFs, DOCX, XLSX
- **FAISS Indexing**: Vector-based search and retrieval
- **CLIP/BLIP Embeddings**: Multimodal data integration

### 🛠️ Utilities
- **Image Processing**: Hashing, classification, deduplication
- **Document Processing**: Text extraction, chunking, metadata handling
- **Query Management**: Text and image queries

---

## 🔗 API Overview

### Document Management
- **Upload**: `POST /upload/document`
- **Process**: `POST /process/documents`
- **List**: `GET /get/documents`
- **Delete**: `DELETE /delete/document`

### Query Processing
- **Text**: `POST /query/text`
- **Image**: `POST /query/image`
- **Reset Chat**: `POST /chat/reset`
- **Chat History**: `GET /chat/history`

For detailed endpoints, consult the [Technical Reference](docs/technical-reference.md#api-documentation).

---

## 🔄 Update Service

### Features
- **Automated Updates**: Monitors GitHub for new releases
- **Backup System**: Ensures data persistence during updates
- **Service Control**: `systemctl` integration

For update procedures, refer to the [Update Service Guide](docs/update-service.md).

---

## 📘 Documentation

1. [Installation Guide](docs/installation.md)
2. [Technical Reference](docs/technical-reference.md)
3. [Frontend Guide](docs/frontend.md)
4. [Models Documentation](docs/models.md)
5. [Utils Reference](docs/utils.md)
6. [Update Service](docs/update-service.md)

---

## 💡 Support
For technical assistance, contact [Mike Kertser](mailto:mikek@atlantium.com).

