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
});

// Add to scripts.js
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadList = document.getElementById('upload-list');
    const processBtn = document.getElementById('process-documents');
    const files = new Set();

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
            if (!files.has(file.name) && isValidFile(file)) {
                files.add(file.name);
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
            <span class="file-name">${file.name}</span>
            <span class="file-remove">×</span>
        `;

        div.querySelector('.file-remove').addEventListener('click', () => {
            div.remove();
            files.delete(file.name);
            if (files.size === 0) {
                processBtn.style.display = 'none';
            }
        });

        return div;
    }

    processBtn.addEventListener('click', async () => {
        processBtn.disabled = true;
        processBtn.textContent = "Processing..."; // Visual feedback

        const uploadPromises = Array.from(uploadList.children).map(async (item) => {
            const fileName = item.querySelector('.file-name').textContent;
            const file = Array.from(fileInput.files).find(f => f.name === fileName);

            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload/document', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Processing failed');
                processBtn.textContent = "Uploaded Successfully";
                processBtn.style.backgroundColor = '#28a745';

                setTimeout(() => {
                    uploadList.innerHTML = '';
                    files.clear();
                    processBtn.style.display = 'none';
                }, 2000);

            } catch (error) {
                processBtn.textContent = "Error - Try Again";
                processBtn.style.backgroundColor = '#dc3545';
                console.error('Processing error:', error);
            }
        });

        await Promise.all(uploadPromises);

        try {
            const response = await fetch('/process/documents', {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Processing failed');

            alert('Documents added to DB successfully!');
            uploadList.innerHTML = '';
            files.clear();
            processBtn.style.display = 'none';
        } catch (error) {
            alert('Error processing documents. Please try again.');
            console.error('Processing error:', error);
        }

        processBtn.disabled = false;
    });
});