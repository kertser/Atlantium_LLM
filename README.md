# Atlantium AI RAG-Based Technical Assistant for UV Systems

Atlantium's Retrieval-Augmented Generation (RAG) system is an advanced AI-driven technical assistant designed for UV system management. This platform combines the capabilities of FAISS indexing, CLIP embeddings, and GPT-based analysis to process text and images for efficient query handling, document management, and image analysis. Below is a comprehensive guide for setting up, running, and maintaining this system.
### _*This application is still under development*_

---

## Features

### Core Capabilities
- **Text and Image Retrieval**:
  - Extracts and analyzes text and images from supported file formats (PDF, DOCX, XLSX).
  - Uses FAISS for high-speed vector-based similarity searches.
- **AI-Powered Responses**:
  - Integrates APIs from OpenAI, Anthropic, and other LLM providers for enhanced query response generation.
- **Image Deduplication and Metadata Handling**:
  - Utilizes perceptual hashing and zero-shot classification for efficient image processing.
- **Dynamic Document Management**:
  - Supports document uploads, indexing, and real-time updates.
- **Photo Analysis and Image Retrieval**:
  - Analyze photos with contextual information and retrieve visually similar images.

---

## System Architecture

The RAG system consists of the following components:

1. **FAISS Indexing**: Efficient storage and retrieval of text and image embeddings.
2. **CLIP Model**: Used for generating embeddings for text and images.
3. **FastAPI Backend**:
   - Manages RESTful APIs for document uploads, query processing, and data management.
   - Integrates with GPT models and other APIs for text-based and image-based query resolution.
4. **Frontend**:
   - Interactive web application for managing documents and engaging with the assistant.
5. **Future Extensions (TBD)**:
   - Fine-tuned LLMs (local and remote).
   - Special UV System Dose Calculator agent for system sizing.
   - Other agents
   - Refined reasoning, implementing CoT and MoA
   - Android application with multimodal capabilities.
   - CI/CD with webhooks from GitHub

---

## Prerequisites

1. **Docker**: Ensure Docker and Docker Compose are installed.
2. **Python**: Requires Python 3.12.
3. **API Keys**: Configure API keys from OpenAI, Anthropic, or other supported providers in the `.env` file.
4. **Hardware**: GPU support is optional but recommended for CLIP and FAISS acceleration.

---

## Installation

### Clone the Repository
```bash
$ git clone https://github.com/kertser/Atlantium_LLM.git
$ cd Atlantium_LLM
```

### Install Dependencies
Use the provided `requirements.txt` file:
```bash
$ pip install -r requirements.txt
```
For GPU acceleration, ensure appropriate versions of PyTorch and FAISS are installed.

### Environment Configuration
Create a `.env` file in the project root with the following:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## Running the System

### First-Time Setup
To initialize the system and index:
```bash
$ INITIALIZE_RAG=true docker-compose up -d
```

### Regular Usage
Start the services:
```bash
$ docker-compose up -d
```
Access the web application at `http://localhost:9000`.

### CLI Upload
Upload documents via CLI:
```bash
$ curl -X POST "http://localhost:9000/upload/document" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/document.pdf"
```

---

## Document Management

### Adding Documents
Place documents in the `Raw Documents` directory or upload via the web interface or CLI. Supported formats include PDF, DOCX, and XLSX.

### Removing Documents
Delete unwanted documents from the `Raw Documents` directory or through the web interface. Ensure the index is updated to reflect the changes.

### Indexing Documents
Run the indexing process manually or use the web interface to ensure newly added or updated documents are processed and retrievable.

---

## Analysis and Image Retrieval

### Photo Analysis
Upload photos with relevant context through the web interface. The system uses CLIP and LLMs to analyze the images, extracting technical details and identifying key elements.

### Image Retrieval
Retrieve visually similar images using FAISS-based similarity searches, enhanced by perceptual hashing and metadata.

---

## Development

### Running Locally
For development and debugging:
```bash
$ python run.py
```

### Testing Document Processing
Use the `rag_system.py` script to process documents:
```bash
$ python RAG_processor.py
```

### Logs
Logs are saved to `system.log`.

---

## Configuration

Configurations are managed in `config.py`. Key settings include:
- **Paths**:
  - RAW_DOCUMENTS_PATH: Directory for uploaded documents.
  - FAISS_INDEX_PATH: Path to the FAISS index file.
- **CLIP Settings**:
  - CLIP_MODEL_NAME: Model used for embeddings.
  - USE_GPU: Set to `True` to enable GPU acceleration.
- **Response Formatting**:
  - DEFAULT_LINE_LENGTH: Line length for responses.
  - BULLET_INDENT: Indentation for bullet points.

---

## Supported File Formats

The system supports the following file types:
- **PDF**
- **Word Documents (.docx)**
- **Excel Files (.xlsx)**

---

## Troubleshooting

### Common Issues

1. **No Documents Found**:
   - Ensure the documents are in the correct directory.
   - Verify supported file formats.

2. **Errors During Processing**:
   - Check `system.log` for detailed error messages.
   - Ensure the FAISS index and metadata files are accessible.

3. **API Key Errors**:
   - Verify that the API keys are correctly set in `.env`.

4. **Docker Issues**:
   - Run `$ docker-compose logs` to identify errors.

---

## Maintenance

### Updating the System
Pull the latest changes and restart services:
```bash
$ git pull origin main
$ docker-compose build
$ docker-compose up -d
```

### Cleaning Up
To clean up containers and volumes:
```bash
$ docker rm -f $(docker ps -a -q)
$ docker system prune --all --volumes --force
```
---

## License

This project is private commercial software. All rights reserved.

---

For additional support, contact [Atlantium Technologies Support](mailto:support@atlantium.com).

