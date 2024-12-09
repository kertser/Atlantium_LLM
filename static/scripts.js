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
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadList = document.getElementById('upload-list');
    const processBtn = document.getElementById('process-documents');

    // Create a Map to store file objects
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
    fileInput.addEventListener('change', (e) => handleFiles([...e.target.files]));

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
        const files = [...e.dataTransfer.files];
        handleFiles(files);
    }

    function isValidFile(file) {
        const validExtensions = ['.pdf', '.docx', '.xlsx'];
        return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
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

    // Initial document load
    loadDocuments();
});