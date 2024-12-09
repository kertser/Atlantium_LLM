document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const input = document.querySelector('textarea');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    const attachImageButton = document.getElementById('send-image');
    attachImageButton.textContent = 'Attach Image';

    let currentAttachedImage = null;

    // Create hidden file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    async function sendMessageWithImage(message, imageFile = null) {
        try {
            const formData = new FormData();
            formData.append('query', message);

            if (imageFile) {
                formData.append('image', imageFile);

                const response = await fetch('/query/image', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                return data.response;
            } else {
                const response = await fetch('/query/text', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (!data.response) {
                    throw new Error('Invalid response format');
                }
                return data.response;
            }
        } catch (error) {
            console.error('Error:', error);
            return {
                text_response: 'Sorry, there was an error processing your request.',
                images: []
            };
        }
    }

    function createImageElement(imageData) {
        const container = document.createElement('div');
        container.className = 'image-container';

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${imageData.image}`;
        img.alt = imageData.caption || 'Response image';
        img.loading = 'lazy';

        const caption = document.createElement('div');
        caption.className = 'image-caption';
        caption.textContent = deduplicateCaption(imageData.caption || '');

        container.appendChild(img);
        container.appendChild(caption);

        // Add click handler for enlarging
        container.onclick = () => showModal(imageData.image);

        return container;
    }

    function showModal(imageBase64) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'flex';

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${imageBase64}`;
        img.className = 'modal-content';

        modal.onclick = (e) => {
            if (e.target === modal) modal.remove();
        };
        modal.appendChild(img);
        document.body.appendChild(modal);
    }

    function handleImageAttachment(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Find the preview container
        const previewContainer = document.querySelector('.attached-image-preview');
        if (!previewContainer) {
            console.error('Preview container not found');
            return;
        }

        // Create preview image
        const previewImage = document.createElement('img');
        previewImage.src = URL.createObjectURL(file);

        const removeButton = document.createElement('button');
        removeButton.className = 'remove-image-button';
        removeButton.innerHTML = '×';
        removeButton.onclick = () => {
            previewContainer.style.display = 'none';
            previewContainer.innerHTML = '';
            currentAttachedImage = null;
            fileInput.value = '';
        };

        // Clear previous preview and add new
        previewContainer.innerHTML = '';
        previewContainer.appendChild(previewImage);
        previewContainer.appendChild(removeButton);
        previewContainer.style.display = 'block';  // Show the container

        currentAttachedImage = file;
    }

    function addMessage(content, isUser = false) {
        console.log('Adding message:', content, 'isUser:', isUser);

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;

        // Add text content with formatting
        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';

        // Apply formatting for assistant messages only
        if (!isUser) {
            textDiv.innerHTML = formatMessageText(content.text_response);
        } else {
            textDiv.textContent = content.text || content;
        }

        messageDiv.appendChild(textDiv);

        // Add image for user message if present
        if (isUser && content.image) {
            const imageDiv = document.createElement('div');
            imageDiv.className = 'image-container';
            const img = document.createElement('img');
            img.src = URL.createObjectURL(content.image);
            imageDiv.appendChild(img);
            messageDiv.appendChild(imageDiv);
        }

        // Add response images if any
        if (!isUser && content.images && content.images.length > 0) {
            console.log('Processing images:', content.images);
            const imageGrid = document.createElement('div');
            imageGrid.className = 'image-grid';

            content.images.forEach(img => {
                try {
                    const imageElement = createImageElement(img);
                    imageGrid.appendChild(imageElement);
                    console.log('Added image to grid');
                } catch (error) {
                    console.error('Error adding image:', error);
                }
            });

            messageDiv.appendChild(imageGrid);
        }

        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function formatMessageText(text) {
        // Format headers
        text = text.replace(/^# (.+)$/gm, '<h3 class="message-header">$1</h3>');

        // Format subheaders
        text = text.replace(/^([A-Za-z].+)$/gm, '<h4 class="message-subheader">$1</h4>');

        // Ensure new lines before bullet points
        text = text.replace(/(.+?)(?=\n*?• |\n*?$)/g, '<p>$1</p>\n');

        // Format bullets
        text = text.replace(/• (.+)/g, '<li class="message-bullet">$1</li>');

        // Wrap bullets in list, ensuring each bullet starts on a new line
        text = text.replace(/(<li[^>]*>.*<\/li>\n?)+/g, (match) => {
            // Split by line breaks and then join with new paragraph tags for each bullet
            return `<ul class="message-list">\n${match.split('\n').map(item => `  ${item}`).join('\n')}\n</ul>`;
        });

        // Format bold text
        text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        return text;
    }

    function deduplicateCaption(caption) {
        // Split by pipe and trim each part
        const parts = caption.split('|').map(p => p.trim());

        // Extract unique source document names without image numbers
        const sources = parts
            .map(part => part.replace(/^Image \d+ from /, ''))
            .filter((value, index, self) => self.indexOf(value) === index);

        return sources.join(' | ');
    }

    async function handleSend() {
        const message = input.value.trim();
        if (message || currentAttachedImage) {
            // Prepare message content
            const messageContent = {
                text: message,
                image: currentAttachedImage
            };

            // Show the message in chat
            addMessage(messageContent, true);

            // Add loading message
            const loadingMessage = addLoadingMessage();

            // Clear the image preview and current image
            const previewContainer = document.querySelector('.attached-image-preview');
            if (previewContainer) {
                previewContainer.style.display = 'none';
                previewContainer.innerHTML = '';
            }
            currentAttachedImage = null;

            // Clear input
            input.value = '';
            input.style.height = 'auto';

            try {
                // Send message and get response
                const response = await sendMessageWithImage(message, currentAttachedImage);

                // Remove loading message
                if (loadingMessage && loadingMessage.parentNode) {
                    loadingMessage.remove();
                }

                // Add the response to chat
                if (typeof response === 'string') {
                    addMessage({ text_response: response, images: [] });
                } else {
                    addMessage(response);
                }
            } catch (error) {
                // Remove loading message
                if (loadingMessage && loadingMessage.parentNode) {
                    loadingMessage.remove();
                }

                // Show error message
                addMessage({
                    text_response: 'Sorry, there was an error processing your request.',
                    images: []
                });
            }
        }
    }

    async function handleReset() {
        try {
            await fetch('/chat/reset', { method: 'POST' });
            chatLog.innerHTML = '';
            input.value = '';
            const previewContainer = document.querySelector('.attached-image-preview');
            if (previewContainer) {
                previewContainer.style.display = 'none';
                previewContainer.innerHTML = '';
            }
            currentAttachedImage = null;
            fileInput.value = '';
        } catch (error) {
            console.error('Error resetting chat:', error);
        }
    }

    function addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-message';
        loadingDiv.id = 'loading-message';

        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';

        const loadingText = document.createElement('span');
        loadingText.textContent = 'Generating response';

        const loadingDots = document.createElement('span');
        loadingDots.className = 'loading-dots';

        textDiv.appendChild(loadingText);
        textDiv.appendChild(loadingDots);
        loadingDiv.appendChild(textDiv);

        chatLog.appendChild(loadingDiv);
        chatLog.scrollTop = chatLog.scrollHeight;

        return loadingDiv;
    }

    function adjustTextareaHeight() {
        input.style.height = 'auto';
        input.style.height = (input.scrollHeight) + 'px';
    }

    async function loadDocuments() {
        try {
            const response = await fetch('/documents');
            const data = await response.json();

            const tbody = document.querySelector('.documents-table tbody');
            document.getElementById('doc-count').textContent = data.documents.length;

            tbody.innerHTML = data.documents.map(doc => `
                <tr>
                    <td>${doc.name}</td>
                    <td>${doc.type}</td>
                    <td>${formatFileSize(doc.size)}</td>
                    <td>${new Date(doc.modified * 1000).toLocaleDateString()}</td>
                </tr>
            `).join('');
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    function formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Byte';
        const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
        return Math.round(100 * (bytes / Math.pow(1024, i))) / 100 + ' ' + sizes[i];
    }

    // Event listeners
    attachImageButton.addEventListener('click', () => {
        fileInput.click();
    });
    fileInput.addEventListener('change', handleImageAttachment);
    sendButton.addEventListener('click', handleSend);
    resetButton.addEventListener('click', handleReset);
    input.addEventListener('input', adjustTextareaHeight);

    // Initial focus
    input.focus();
    loadDocuments();
});

async function loadDocuments() {
    try {
        const response = await fetch('/get/documents');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const documents = await response.json();

        // Update document count
        const docCount = document.getElementById('doc-count');
        if (docCount) {
            docCount.textContent = documents.length;
        }

        // Update documents table
        const tableWrapper = document.querySelector('.documents-table-wrapper');
        if (tableWrapper) {
            const table = document.createElement('table');
            table.className = 'documents-table';

            // Create table header
            const thead = document.createElement('thead');
            thead.innerHTML = `
                <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Size</th>
                    <th>Modified</th>
                </tr>
            `;
            table.appendChild(thead);

            // Create table body
            const tbody = document.createElement('tbody');
            tbody.innerHTML = documents.map(doc => `
                <tr>
                    <td>${escapeHtml(doc.name)}</td>
                    <td>${escapeHtml(doc.type)}</td>
                    <td>${formatFileSize(doc.size)}</td>
                    <td>${formatDate(doc.modified)}</td>
                </tr>
            `).join('');
            table.appendChild(tbody);

            // Clear and update table
            tableWrapper.innerHTML = '';
            tableWrapper.appendChild(table);

            console.log(`Updated document list with ${documents.length} documents`);
        }
    } catch (error) {
        console.error('Error loading documents:', error);
        // Optionally show error to user
        const tableWrapper = document.querySelector('.documents-table-wrapper');
        if (tableWrapper) {
            tableWrapper.innerHTML = '<div class="error-message">Error loading documents. Please try again.</div>';
        }
    }
}

// Helper functions

// Helper functions
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadList = document.getElementById('upload-list');
    const processBtn = document.getElementById('process-documents');
    const fileMap = new Map();

    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight);
    });

    dropZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFiles);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const newFiles = [...dt.files];
        handleFiles({ target: { files: newFiles } });
    }

    function handleFiles(e) {
        const newFiles = [...e.target.files];
        newFiles.forEach(file => {
            if (isValidFile(file)) {
                fileMap.set(file.name, file); // Store the actual file object
                uploadList.appendChild(createFileItem(file));
                processBtn.style.display = 'block';
            }
        });
    }

    function isValidFile(file) {
        const validExtensions = ['.pdf', '.docx', '.xlsx'];
        return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    }

    function createFileItem(file) {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.innerHTML = `
            <span class="file-name">${escapeHtml(file.name)}</span>
            <span class="file-remove">×</span>
        `;

        div.querySelector('.file-remove').addEventListener('click', () => {
            div.remove();
            fileMap.delete(file.name);
            if (fileMap.size === 0) {
                processBtn.style.display = 'none';
            }
        });

        return div;
    }

    processBtn.addEventListener('click', async () => {
        try {
            processBtn.disabled = true;
            processBtn.textContent = "Uploading Documents...";
            processBtn.classList.add('processing');

            // Upload all files from the fileMap
            const uploadPromises = Array.from(fileMap.values()).map(async (file) => {
                const formData = new FormData();
                formData.append('file', file);

                const uploadResponse = await fetch('/upload/document', {
                    method: 'POST',
                    body: formData
                });

                if (!uploadResponse.ok) {
                    const errorData = await uploadResponse.json();
                    throw new Error(errorData.detail || `Failed to upload ${file.name}`);
                }

                return uploadResponse.json();
            });

            // Wait for all uploads to complete
            await Promise.all(uploadPromises);
            console.log('All files uploaded successfully');

            // Process the documents
            processBtn.textContent = "Processing Documents...";
            const processResponse = await fetch('/process/documents', {
                method: 'POST'
            });

            if (!processResponse.ok) {
                const errorData = await processResponse.json();
                throw new Error(errorData.detail || 'Document processing failed');
            }

            const result = await processResponse.json();

            // Success handling
            processBtn.textContent = "Processing Complete";
            processBtn.classList.remove('processing');
            processBtn.style.backgroundColor = '#28a745';

            // Clear upload list
            uploadList.innerHTML = '';
            fileMap.clear();

            // Refresh document list
            await loadDocuments();

            // Reset button after delay and hide it
            setTimeout(() => {
                processBtn.textContent = "Process Documents";
                processBtn.style.backgroundColor = '';
                processBtn.disabled = false;
                processBtn.style.display = 'none'; // Hide the button
            }, 3000);

        } catch (error) {
            console.error('Processing error:', error);
            processBtn.textContent = "Process Documents";
            processBtn.classList.remove('processing');
            processBtn.style.backgroundColor = '#dc3545';
            processBtn.disabled = false;
            alert(`Error: ${error.message}`);
        }
    });
});