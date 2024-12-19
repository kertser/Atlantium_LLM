# Frontend Documentation

## Overview

The frontend provides a web interface for document management, chat interaction, and system control.

## Components

### Static Files

```
static/
├── index.html    # Main application page
├── styles.css    # Application styling
├── scripts.js    # Client-side functionality
└── favicon.png   # Web-app icon
```

### Key Features

1. **Document Management**
   - Upload interface
   - Folder navigation
   - Document operations (open, download, delete)

2. **Chat Interface**
   - Message history
   - Image attachments
   - Markdown rendering

3. **System Controls**
   - Document processing
   - System reset
   - File and Folder management

## Implementation Details

### Document Upload
```javascript
async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('folder', currentFolderPath);
    // ... implementation details
}
```

### Chat Functionality
```javascript
function formatMessageText(text) {
    // Handles markdown formatting
    // Processes math content
    // Formats lists and headers
}

async function handleSend() {
    // Processes messages
    // Handles image attachments
    // Updates chat display
}
```

## Integration Points

1. **API Endpoints**
   - Document upload/download
   - Chat interaction
   - System control

2. **WebSocket Communication**
   - Real-time updates
   - Processing status
