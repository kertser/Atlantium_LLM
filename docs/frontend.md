# Frontend Documentation

## Overview

The frontend implements several key functional pipelines for document management and chat interaction:

### Document Processing Pipeline
1. Document Upload
   - User uploads files through drag-drop or file selector
   - Files are validated (type, size, count limits)
   - Files are uploaded to server via `/upload/document` endpoint
   - Server processes documents using RAG system

2. Document Management
   - Files are organized in hierarchical folder structure
   - Each file/folder supports context menu operations
   - Bulk operations available through multi-select
   - Real-time document count and status updates

### Chat Interaction Pipeline
1. Text Queries
   - User input is processed for special characters and formatting
   - Queries are sent to server for RAG-enhanced processing
   - Responses include text and relevant images
   - Chat history is maintained and can be reset

2. Image Queries
   - Images can be attached to messages
   - Preview system shows attached images
   - Images are processed with optional text queries
   - Responses include image analysis and relevant documentation

### Authentication Flow
1. Protected Operations
   - Document deletion
   - Folder management
   - System initialization
   - Batch operations
2. Authentication Modal
   - Password-based verification
   - Session-based authentication state
   - Protected operation handling

## Core Components

The frontend provides a web-based interface for document management, chat interaction, and system control, implemented using vanilla JavaScript for maximum compatibility.

## Core Components

### Static Files Structure
```
static/
├── index.html       # Main application page
├── styles.css       # Application styling
├── scripts.js       # Client-side functionality
├── logo_long.jpg    # Application logo
└── favicon.png      # Application icon
```

## Component Architecture

### Document Management
```javascript
// File Upload
const handleFiles = (files) => {
    if (fileMap.size + files.length > maxfiles) {
        alert(`Maximum ${maxfiles} files allowed`);
        return;
    }
    // Process files...
}

// Document List
const loadDocuments = async (currentPath = '') => {
    // Load and display documents...
}

// Document Operations
const createFolder = async (parentPath, folderName) => {
    // Create new folder...
}

const deleteDocument = async (path) => {
    // Delete document...
}
```

### Chat Interface
```javascript
// Message Handling
const addMessage = (content, isUser = false) => {
    // Add message to chat...
}

// Query Processing
const sendMessageWithImage = async (message, imageFile = null) => {
    // Process and send message...
}

// Chat Reset
const handleReset = async () => {
    // Reset chat history...
}
```

### Image Management
```javascript
// Image Preview
const handleImageAttachment = (event) => {
    // Handle image preview...
}

// Image Upload
const processImageUpload = async (file) => {
    // Process image upload...
}
```

## Event Handlers

### Document Events
```javascript
// Drag and Drop
dropZone.addEventListener('drop', handleDrop);
dropZone.addEventListener('dragover', preventDefaults);

// File Selection
fileInput.addEventListener('change', handleFiles);

// Context Menu
const createContextMenu = (e, fileName, filePath) => {
    // Create context menu...
}
```

### Chat Events
```javascript
// Message Input
input.addEventListener('input', adjustTextareaHeight);

// Send Button
sendButton.addEventListener('click', handleSend);

// Reset Button
resetButton.addEventListener('click', handleReset);
```

## API Integration

### Document Endpoints
```javascript
// Upload Document
const uploadDocument = async (file, folder = '') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('folder', folder);
    
    const response = await fetch('/upload/document', {
        method: 'POST',
        body: formData
    });
    // Handle response...
}

// Process Documents
const processDocuments = async () => {
    const response = await fetch('/process/documents', {
        method: 'POST'
    });
    // Handle response...
}
```

### Query Endpoints
```javascript
// Text Query
const sendTextQuery = async (query) => {
    const formData = new FormData();
    formData.append('query', query);
    
    const response = await fetch('/query/text', {
        method: 'POST',
        body: formData
    });
    // Handle response...
}

// Image Query
const sendImageQuery = async (image, query = null) => {
    const formData = new FormData();
    formData.append('image', image);
    if (query) formData.append('query', query);
    
    const response = await fetch('/query/image', {
        method: 'POST',
        body: formData
    });
    // Handle response...
}
```

## UI Components

### Message Display
```javascript
const formatMessageText = (text) => {
    // Format message content...
}

const createImageElement = (imageData) => {
    // Create image element...
}
```

### Document List
```javascript
const updateFileCount = () => {
    // Update file counter...
}

const createFileItem = (file) => {
    // Create file list item...
}
```

## Error Handling

### Common Patterns
1. Network Errors
2. File Size Limits
3. Invalid File Types
4. Authentication Errors
5. Server Response Errors

### Error Display
```javascript
const showError = (message) => {
    // Display error message...
}

const handleAPIError = async (response) => {
    // Handle API errors...
}
```

## Authentication

```javascript
const createAuthModal = async () => {
    // Create authentication modal...
}

const checkAuthentication = async () => {
    // Check authentication status...
}
```

## CSS Structure

### Core Components
```css
/* Layout */
.app-container { ... }
.chat-container { ... }
.documents-container { ... }

/* Messages */
.message { ... }
.user-message { ... }
.assistant-message { ... }

/* Documents */
.upload-container { ... }
.file-item { ... }
.folder-row { ... }
```

## Related Documentation

- [Technical Reference](../docs/technical-reference.md)
- [Installation Guide](../docs/installation.md)
- [Models Documentation](../docs/models.md)
- [Utils Documentation](../docs/utils.md)