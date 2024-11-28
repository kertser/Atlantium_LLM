document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const input = document.querySelector('textarea');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');

    async function sendMessage(message) {
        try {
            const formData = new FormData();
            formData.append('query', message);

            console.log('Sending query:', message);

            const response = await fetch('/query/text', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received response:', data);

            if (!data.response) {
                console.error('Invalid response format:', data);
                throw new Error('Invalid response format');
            }

            return data.response;
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

    function addMessage(content, isUser = false) {
        console.log('Adding message:', content, 'isUser:', isUser);

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;

        // Add text content
        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';
        textDiv.textContent = isUser ? content : content.text_response;
        messageDiv.appendChild(textDiv);

        // Add images if any
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

        const chatLog = document.getElementById('chat-log');
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function handleSend() {
        const message = input.value.trim();
        if (message) {
            addMessage(message, true);
            input.value = '';
            input.style.height = 'auto';

            const response = await sendMessage(message);
            addMessage(response);
        }
    }

    async function handleReset() {
        try {
            await fetch('/chat/reset', { method: 'POST' });
            chatLog.innerHTML = '';
            input.value = '';
        } catch (error) {
            console.error('Error resetting chat:', error);
        }
    }

    function adjustTextareaHeight() {
        input.style.height = 'auto';
        input.style.height = (input.scrollHeight) + 'px';
    }

    sendButton.addEventListener('click', handleSend);
    resetButton.addEventListener('click', handleReset);

    input.addEventListener('input', adjustTextareaHeight);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Initial focus
    input.focus();
});