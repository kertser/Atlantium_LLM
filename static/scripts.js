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

        if (imageData.caption) {
            const caption = document.createElement('div');
            caption.className = 'image-caption';
            caption.textContent = imageData.caption;
            container.appendChild(caption);
        }

        container.appendChild(img);
        return container;
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
        // Format headers (# text)
        text = text.replace(/^# (.+)$/gm, '<h3 class="message-header">$1</h3>');

        // Format subheaders (lines that match specific patterns)
        text = text.replace(/^([A-Za-z]+(?: and [A-Za-z]+| Settings))$/gm, '<h4 class="message-subheader">$1</h4>');

        // Format bold text within lines using <strong>
        text = text.replace(/\*\*(.+?)\*\*/gm, '<strong>$1</strong>');

        // Format bullet points
        text = text.replace(/^\* (.+)$/gm, '<li class="message-bullet"><strong>$1</strong></li>');

        // Wrap consecutive bullet points into a list
        text = text.replace(
            /(<li class="message-bullet"><strong>.+<\/strong>\n?)+/g,
            match => `<ul class="message-list">${match}</ul>`
        );

        return text;
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