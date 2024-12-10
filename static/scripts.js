// Helper functions (defined outside DOMContentLoaded to be available globally)
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
        const tableWrapper = document.querySelector('.documents-table-wrapper');
        if (tableWrapper) {
            tableWrapper.innerHTML = '<div class="error-message">Error loading documents. Please try again.</div>';
        }
    }
}

// Main event listener
document.addEventListener('DOMContentLoaded', () => {
    // Initialize document upload elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadList = document.getElementById('upload-list');
    const processBtn = document.getElementById('process-documents');
    const fileMap = new Map();

    // Initialize chat elements
    const chatLog = document.getElementById('chat-log');
    const input = document.querySelector('textarea');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    const attachImageButton = document.getElementById('send-image');
    let currentAttachedImage = null;

    // Create hidden file input for chat images
    const chatImageInput = document.createElement('input');
    chatImageInput.type = 'file';
    chatImageInput.accept = 'image/*';
    chatImageInput.style.display = 'none';
    document.body.appendChild(chatImageInput);

    // Document Upload Functions
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

    function isValidFile(file) {
        const validExtensions = ['.pdf', '.docx', '.xlsx'];
        return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    }

    function handleDrop(e) {
        const files = [...e.dataTransfer.files];
        handleFiles(files);
    }

    function handleFiles(files) {
        files.forEach(file => {
            if (isValidFile(file)) {
                if (fileMap.has(file.name)) {
                    alert(`File "${file.name}" has already been added`);
                    return;
                }
                fileMap.set(file.name, file);
                uploadList.appendChild(createFileItem(file));
                processBtn.style.display = 'block';
            }
        });
    }

    function createFileItem(file) {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.innerHTML = `
            <span class="file-name">${escapeHtml(file.name)}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
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

    // Chat Functions
    function formatMessageText(text) {
        text = text.replace(/^# (.+)$/gm, '<h3 class="message-header">$1</h3>');
        text = text.replace(/^([A-Za-z].+)$/gm, '<h4 class="message-subheader">$1</h4>');
        text = text.replace(/(.+?)(?=\n*?• |\n*?$)/g, '<p>$1</p>\n');
        text = text.replace(/• (.+)/g, '<li class="message-bullet">$1</li>');
        text = text.replace(/(<li[^>]*>.*<\/li>\n?)+/g,
            match => `<ul class="message-list">\n${match.split('\n').map(item => `  ${item}`).join('\n')}\n</ul>`);
        text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        return text;
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

    function deduplicateCaption(caption) {
        const parts = caption.split('|').map(p => p.trim());
        const sources = parts
            .map(part => part.replace(/^Image \d+ from /, ''))
            .filter((value, index, self) => self.indexOf(value) === index);
        return sources.join(' | ');
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
        // Scroll after adding loading message
        scrollChatToBottom();

        return loadingDiv;
    }

    function adjustTextareaHeight() {
        input.style.height = 'auto';
        input.style.height = (input.scrollHeight) + 'px';
    }

    function scrollChatToBottom() {
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;

        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';

        if (!isUser) {
            textDiv.innerHTML = formatMessageText(content.text_response);
        } else {
            textDiv.textContent = content.text || content;
        }

        messageDiv.appendChild(textDiv);

        if (isUser && content.image) {
            const imageDiv = document.createElement('div');
            imageDiv.className = 'image-container';
            const img = document.createElement('img');
            img.src = URL.createObjectURL(content.image);
            imageDiv.appendChild(img);
            messageDiv.appendChild(imageDiv);
        }

        if (!isUser && content.images && content.images.length > 0) {
            const imageGrid = document.createElement('div');
            imageGrid.className = 'image-grid';

            content.images.forEach(img => {
                const imageElement = createImageElement(img);
                imageGrid.appendChild(imageElement);
            });

            messageDiv.appendChild(imageGrid);
        }

        chatLog.appendChild(messageDiv);

        // Handle image loading
        const images = messageDiv.getElementsByTagName('img');
        if (images.length > 0) {
            let loadedImages = 0;
            const totalImages = images.length;

            const checkAllImagesLoaded = () => {
                loadedImages++;
                if (loadedImages === totalImages) {
                    scrollChatToBottom();
                }
            };

            Array.from(images).forEach(img => {
                if (img.complete) {
                    checkAllImagesLoaded();
                } else {
                    img.onload = checkAllImagesLoaded;
                    img.onerror = checkAllImagesLoaded;
                }
            });
        }

        // Always scroll immediately after adding the message
        scrollChatToBottom();
    }

    // Event Handlers
    async function handleImageAttachment(event) {
        const file = event.target.files[0];
        if (!file) return;

        const previewContainer = document.querySelector('.attached-image-preview');
        if (!previewContainer) {
            console.error('Preview container not found');
            return;
        }

        const previewImage = document.createElement('img');
        previewImage.src = URL.createObjectURL(file);

        const removeButton = document.createElement('button');
        removeButton.className = 'remove-image-button';
        removeButton.innerHTML = '×';
        removeButton.onclick = () => {
            previewContainer.style.display = 'none';
            previewContainer.innerHTML = '';
            currentAttachedImage = null;
            chatImageInput.value = '';
        };

        previewContainer.innerHTML = '';
        previewContainer.appendChild(previewImage);
        previewContainer.appendChild(removeButton);
        previewContainer.style.display = 'block';

        currentAttachedImage = file;
    }

    async function handleSend() {
        const message = input.value.trim();
        if (message || currentAttachedImage) {
            const messageContent = {
                text: message,
                image: currentAttachedImage
            };

            // Clear input and preview first
            const previewContainer = document.querySelector('.attached-image-preview');
            if (previewContainer) {
                previewContainer.style.display = 'none';
                previewContainer.innerHTML = '';
            }
            currentAttachedImage = null;
            input.value = '';
            input.style.height = 'auto';

            // Add user message
            addMessage(messageContent, true);

            // Add loading message
            const loadingMessage = addLoadingMessage();

            try {
                // Get response from server
                const response = await sendMessageWithImage(message, currentAttachedImage);

                // Remove loading message
                if (loadingMessage && loadingMessage.parentNode) {
                    loadingMessage.remove();
                }

                // Add response message
                if (typeof response === 'string') {
                    addMessage({ text_response: response, images: [] });
                } else {
                    addMessage(response);
                }

            } catch (error) {
                if (loadingMessage && loadingMessage.parentNode) {
                    loadingMessage.remove();
                }
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
            chatImageInput.value = '';
        } catch (error) {
            console.error('Error resetting chat:', error);
        }
    }

    // API Functions
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

    // Document Upload Event Listeners
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
    fileInput.addEventListener('change', (e) => handleFiles([...e.target.files]));

    // Chat Event Listeners
    attachImageButton.addEventListener('click', () => chatImageInput.click());
    chatImageInput.addEventListener('change', handleImageAttachment);
    sendButton.addEventListener('click', handleSend);
    resetButton.addEventListener('click', handleReset);
    input.addEventListener('input', adjustTextareaHeight);

    // Process Button Event Handler
    processBtn.addEventListener('click', async () => {
        if (fileMap.size === 0) {
            alert('Please add some files first');
            return;
        }

        try {
            processBtn.disabled = true;
            processBtn.textContent = "Uploading Files...";
            processBtn.classList.add('processing');

            // Upload all files from the fileMap
            const uploadPromises = Array.from(fileMap.values()).map(async (file) => {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/upload/document', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || `Failed to upload ${file.name}`);
                }

                return response.json();
            });

            await Promise.all(uploadPromises);

            // Process documents
            processBtn.textContent = "Processing Documents...";
            const processResponse = await fetch('/process/documents', {
                method: 'POST'
            });

            if (!processResponse.ok) {
                const error = await processResponse.json();
                throw new Error(error.detail || 'Processing failed');
            }

            const result = await processResponse.json();

            // Success handling
            processBtn.textContent = "Processing Complete";
            processBtn.classList.remove('processing');
            processBtn.style.backgroundColor = '#28a745';

            // Clean up
            uploadList.innerHTML = '';
            fileMap.clear();

            // Refresh documents list
            await loadDocuments();

            // Reset button after delay
            setTimeout(() => {
                processBtn.textContent = "Process Documents";
                processBtn.style.backgroundColor = '';
                processBtn.disabled = false;
                processBtn.style.display = 'none';
            }, 3000);

        } catch (error) {
            console.error('Processing error:', error);
            processBtn.textContent = "Process Documents";
            processBtn.classList.remove('processing');
            processBtn.style.backgroundColor = '#dc3545';
            processBtn.disabled = false;

            // Reset error state after delay
            setTimeout(() => {
                processBtn.style.backgroundColor = '';
            }, 3000);

            alert(`Error: ${error.message}`);
        }
    });

    // Initial setup
    loadDocuments();
    input.focus();
});