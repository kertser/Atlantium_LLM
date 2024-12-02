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

        // Add text content
        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';
        textDiv.textContent = isUser ? content.text || content : content.text_response;
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

            // Send message and get response
            const response = await sendMessageWithImage(message, currentAttachedImage);

            // Add the response to chat
            if (typeof response === 'string') {
                addMessage({ text_response: response, images: [] });
            } else {
                addMessage(response);
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