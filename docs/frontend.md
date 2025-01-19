# Frontend Documentation

## Overview

The frontend implements several key functional pipelines for document management and chat interaction:

### Document Processing Pipeline

#### Document Upload
- Users upload files through drag-and-drop or a file selector.
- Files are validated based on type, size, and count limits.
- Files are uploaded to the server via the `/upload/document` endpoint.
- The server processes documents using the RAG system.

#### Document Management
- Files are organized in a hierarchical folder structure.
- Each file/folder supports context menu operations.
- Bulk operations are available through multi-select.
- Real-time document count and status updates are displayed.

### Chat Interaction Pipeline

#### Text Queries
- User input is processed for special characters and formatting.
- Queries are sent to the server for RAG-enhanced processing.
- Responses include text and relevant images.
- Chat history is maintained and can be reset.

#### Image Queries
- Images can be attached to messages.
- A preview system shows attached images.
- Images are processed with optional text queries.
- Responses include image analysis and relevant documentation.

### Authentication Flow

#### Protected Operations
- Document deletion
- Folder management
- System initialization
- Batch operations

#### Authentication Modal
- Password-based verification
- Session-based authentication state
- Handling of protected operations

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

### Document Management Functions
- **handleFiles()**: Processes file uploads with size and count validation.
- **loadDocuments()**: Retrieves and displays the document list.
- **createFolder()**: Creates new folders in the hierarchy.
- **deleteDocument()**: Removes documents from the system.
- **validatePath()**: Ensures valid file/folder paths.
- **moveDocument()**: Handles document relocation.
- **renameDocument()**: Manages document renaming.

### Chat Interface Functions
- **addMessage()**: Adds messages to the chat display.
- **sendMessageWithImage()**: Processes messages with optional images.
- **handleReset()**: Clears chat history.
- **adjustTextareaHeight()**: Manages input field size.
- **formatResponse()**: Formats chat responses.

### Image Management Functions
- **handleImageAttachment()**: Manages image previews.
- **processImageUpload()**: Handles image processing.
- **validateImage()**: Checks image requirements.
- **createImagePreview()**: Generates image previews.

## Event Handlers

### Document Events
- Drag-and-drop zone handling
- File input change detection
- Context menu creation and handling
- Folder navigation

### Chat Events
- Message input handling
- Send button functionality
- Reset button operations
- Image attachment events

## API Integration

### Document Endpoints
- **POST /upload/document**: Uploads new documents.
- **POST /process/documents**: Processes uploaded files.
- **GET /documents/list**: Retrieves the document list.
- **DELETE /documents/delete**: Removes a document.
- **PUT /documents/move**: Relocates a document.
- **PUT /documents/rename**: Renames a document.

### Query Endpoints
- **POST /query/text**: Processes text queries.
- **POST /query/image**: Processes image queries.
- **POST /query/reset**: Resets chat history.
- **WebSocket /query/stream**: Streams query responses.

### Authentication Endpoints
- **POST /auth/verify**: Verifies user credentials.
- **GET /auth/status**: Retrieves authentication status.
- **POST /auth/logout**: Ends the user session.

## UI Components

### Message Display
- Text formatting with markdown support
- Code block highlighting
- Image rendering with zoom capability
- Response streaming indicators

### Document List
- Hierarchical folder view
- File type indicators
- Selection highlighting
- Status indicators
- Context menu integration

## Error Handling

### Common Error Scenarios
1. Network Errors:
   - Connection timeouts
   - Server unavailable
   - Rate limiting
2. File Operations:
   - Size limit exceeded
   - Invalid file types
   - Storage capacity reached
   - Permission denied
3. Authentication:
   - Invalid credentials
   - Session expired
   - Unauthorized access
4. Processing Errors:
   - Document conversion failed
   - Image processing failed
   - Query processing timeout

### Error Display Functions
- **showError()**: Displays error messages.
- **handleAPIError()**: Processes API error responses.
- **logError()**: Records errors for debugging.
- **recoverFromError()**: Implements error recovery.

## CSS Structure

### Layout Components
```css
.app-container { ... }
.chat-container { ... }
.documents-container { ... }
.auth-modal { ... }
```

### Message Styling
```css
.message { ... }
.user-message { ... }
.assistant-message { ... }
.code-block { ... }
```

### Document List Styling
```css
.upload-container { ... }
.file-item { ... }
.folder-row { ... }
.context-menu { ... }
```

## Related Documentation

- [Technical Reference](../docs/technical-reference.md)
- [Installation Guide](../docs/installation.md)
- [Models Documentation](../docs/models.md)
- [Utils Documentation](../docs/utils.md)

