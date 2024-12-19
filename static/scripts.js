// Add current path tracking
let currentFolderPath = '';
let activeContextMenu = null;

// Helper functions (defined outside DOMContentLoaded to be available globally)
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Modify the createContextMenu function in scripts.js
function createContextMenu(e, fileName, filePath) {
    e.preventDefault();
    removeContextMenu();

    const fileExtension = fileName.split('.').pop().toLowerCase();
    const contextMenu = document.createElement('div');
    contextMenu.className = 'context-menu';

    const menuItems = [];

    // Configure menu items based on file type
    if (fileExtension === 'pdf') {
        menuItems.push({
            label: 'Open',
            icon: `<svg viewBox="0 0 24 24" width="16" height="16">
                     <path fill="currentColor" d="M14 3v2H4v13.385L5.763 17H20v-7h2v8a1 1 0 0 1-1 1H5.105L2 22.5V4a1 1 0 0 1 1-1h11zm5 0V0h2v3h3v2h-3v3h-2V5h-3V3h3z"/>
                   </svg>`,
            action: async () => {
                // Existing open action
            }
        });
    }

    // Add download and delete for all file types
    menuItems.push(
        {
            label: 'Download',
            icon: `<svg viewBox="0 0 24 24" width="16" height="16">
                     <path fill="currentColor" d="M3 19h18v2H3v-2zm10-5.828L19.071 7.1l1.414 1.414L12 17 3.515 8.515 4.929 7.1 11 13.17V2h2v11.172z"/>
                   </svg>`,
            action: async () => {
                // Existing download action
            }
        },
        {
            label: 'Delete',
            icon: `<svg viewBox="0 0 24 24" width="16" height="16">
                     <path fill="currentColor" d="M7 4V2h10v2h5v2h-2v15a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6H2V4h5z"/>
                   </svg>`,
            action: async () => {
                // Existing delete action
            }
        }
    );

    // Create menu items
    menuItems.forEach((item, index) => {
        if (index > 0) {
            const separator = document.createElement('div');
            separator.className = 'context-menu-separator';
            contextMenu.appendChild(separator);
        }

        const menuItem = document.createElement('div');
        menuItem.className = 'context-menu-item';
        menuItem.innerHTML = `${item.icon}<span>${item.label}</span>`;
        menuItem.onclick = () => {
            item.action();
            removeContextMenu();
        };
        contextMenu.appendChild(menuItem);
    });

    // Position the menu at the file name element
    const fileNameElement = e.target;
    const rect = fileNameElement.getBoundingClientRect();
    contextMenu.style.left = `${rect.left}px`;
    contextMenu.style.top = `${rect.bottom}px`;

    // Ensure menu doesn't go off screen
    document.body.appendChild(contextMenu);
    const menuRect = contextMenu.getBoundingClientRect();

    if (menuRect.right > window.innerWidth) {
        contextMenu.style.left = `${window.innerWidth - menuRect.width - 5}px`;
    }
    if (menuRect.bottom > window.innerHeight) {
        contextMenu.style.top = `${rect.top - menuRect.height}px`;
    }

    activeContextMenu = contextMenu;

    // Close menu when clicking outside
    setTimeout(() => {
        document.addEventListener('click', removeContextMenu);
        document.addEventListener('contextmenu', removeContextMenu);
    }, 0);
}

function removeContextMenu() {
    if (activeContextMenu && activeContextMenu.parentElement) {
        activeContextMenu.parentElement.removeChild(activeContextMenu);
        activeContextMenu = null;
    }
    document.removeEventListener('click', removeContextMenu);
    document.removeEventListener('contextmenu', removeContextMenu);
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

function getParentPath(path) {
    const parts = path.split('/');
    parts.pop();
    return parts.join('/');
}

// Modal component
function createModal(title, content, onConfirm, onCancel) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>${title}</h3>
            </div>
            <div class="modal-body">
                ${content}
            </div>
            <div class="modal-footer">
                <button class="icon-button" data-action="cancel">Cancel</button>
                <button class="icon-button" data-action="confirm">Confirm</button>
            </div>
        </div>
    `;

    // Add Enter key handler
    modal.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            onConfirm();
            document.body.removeChild(modal);
        }
    });

    modal.querySelector('[data-action="confirm"]').onclick = () => {
        onConfirm();
        document.body.removeChild(modal);
    };

    modal.querySelector('[data-action="cancel"]').onclick = () => {
        onCancel?.();
        document.body.removeChild(modal);
    };

    document.body.appendChild(modal);
    return modal;
}

// Update folder context menu
function createFolderContextMenu(e, folderPath, folderName) {
    e.preventDefault();

    removeContextMenu(); // Remove any existing context menu

    const contextMenu = document.createElement('div');
    contextMenu.className = 'folder-context-menu';

    const menuItems = [
        {
            label: 'Rename',
            icon: `
                <svg viewBox="0 0 24 24" width="16" height="16">
                    <path fill="currentColor" d="M16.293 2.293l3.414 3.414-1.414 1.414-3.414-3.414 1.414-1.414zM13.172 5.414L3 15.586V19h3.414l10.172-10.172-3.414-3.414z"/>
                </svg>
            `,
            action: () => {
                const modal = createModal(
                    'Rename Folder',
                    `
                        <input type="text" class="modal-input" value="${folderName}" 
                               maxlength="255" autofocus>
                    `,
                    async () => {
                        const input = modal.querySelector('.modal-input');
                        const newName = input.value.trim();

                        if (!newName || newName === folderName) return;

                        try {
                            const response = await fetch('/folder/rename', {
                                method: 'PUT',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    path: folderPath,
                                    new_name: newName
                                })
                            });

                            if (!response.ok) {
                                const error = await response.json();
                                throw new Error(error.detail);
                            }

                            await loadDocuments(currentFolderPath);
                        } catch (error) {
                            console.error('Rename folder error:', error);
                            alert(error.message || 'Failed to rename folder');
                        }
                    }
                );
            }
        },
        {
            label: 'Delete',
            icon: `
                <svg viewBox="0 0 24 24" width="16" height="16">
                    <path fill="currentColor" d="M7 4V2h10v2h5v2h-2v15a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6H2V4h5zM6 6v14h12V6H6zm3 3h2v8H9V9zm4 0h2v8h-2V9z"/>
                </svg>
            `,
            className: 'delete',
            action: () => {
                const modal = createModal(
                    'Delete Folder',
                    `
                        <p>Are you sure you want to delete the folder "${folderName}" and all its contents?</p>
                        <p style="color: #dc3545; margin-top: 0.5rem;">This action cannot be undone.</p>
                    `,
                    async () => {
                        try {
                            const response = await fetch(`/folder/delete?path=${encodeURIComponent(folderPath)}`, {
                                method: 'DELETE'
                            });

                            if (!response.ok) {
                                const error = await response.json();
                                throw new Error(error.detail.message || 'Delete failed');
                            }

                            const result = await response.json();
                            if (result.errors && result.errors.length > 0) {
                                console.warn('Delete completed with errors:', result.errors);
                            }

                            await loadDocuments(currentFolderPath);
                        } catch (error) {
                            console.error('Delete folder error:', error);
                            alert(error.message || 'Failed to delete folder');
                        }
                    }
                );
            }
        }
    ];

    // Create menu items
    menuItems.forEach((item, index) => {
        if (index > 0) {
            const separator = document.createElement('div');
            separator.className = 'folder-context-menu-separator';
            contextMenu.appendChild(separator);
        }

        const menuItem = document.createElement('div');
        menuItem.className = `folder-context-menu-item${item.className ? ' ' + item.className : ''}`;
        menuItem.innerHTML = `${item.icon}<span>${item.label}</span>`;
        menuItem.onclick = () => {
            item.action();
            removeContextMenu();
        };
        contextMenu.appendChild(menuItem);
    });

    // Position the menu
    contextMenu.style.left = `${e.pageX}px`;
    contextMenu.style.top = `${e.pageY}px`;

    // Ensure menu doesn't go off screen
    document.body.appendChild(contextMenu);
    const menuRect = contextMenu.getBoundingClientRect();

    if (menuRect.right > window.innerWidth) {
        contextMenu.style.left = `${window.innerWidth - menuRect.width - 5}px`;
    }
    if (menuRect.bottom > window.innerHeight) {
        contextMenu.style.top = `${window.innerHeight - menuRect.height - 5}px`;
    }

    activeContextMenu = contextMenu;

    // Close menu when clicking outside
    setTimeout(() => {
        document.addEventListener('click', removeContextMenu);
        document.addEventListener('contextmenu', removeContextMenu);
    }, 0);
}

function updateFolderContextMenuHandlers() {
    const folderRows = document.querySelectorAll('.folder-row');
    folderRows.forEach(row => {
        const folderNameElement = row.querySelector('.folder-name');
        if (!folderNameElement) return;

        row.addEventListener('contextmenu', (e) => {
            const path = decodeURIComponent(row.dataset.path);
            const name = folderNameElement.textContent.trim();
            if (name !== '..') { // Don't show context menu for parent folder
                createFolderContextMenu(e, path, name);
            }
        });
    });
}

// Update the document context menu to include the system open functionality
function updateDocumentContextMenu() {
    const menuItems = [
        {
            label: 'Open',
            icon: `
                <svg viewBox="0 0 24 24" width="16" height="16">
                    <path fill="currentColor" d="M14 3v2H4v13.385L5.763 17H20v-7h2v8a1 1 0 0 1-1 1H5.105L2 22.5V4a1 1 0 0 1 1-1h11zm5 0V0h2v3h3v2h-3v3h-2V5h-3V3h3z"/>
                </svg>
            `,
            action: async (filePath) => {
                try {
                    const response = await fetch('/open/document', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ path: filePath })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail);
                    }
                } catch (error) {
                    console.error('Open file error:', error);
                    alert(error.message || 'Failed to open file');
                }
            }
        },
        // ... [existing menu items for download and delete]
    ];

    // Update your existing createContextMenu function to use this menuItems array
}

async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('folder', currentFolderPath); // Add current folder path

    const response = await fetch('/upload/document', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `Failed to upload ${file.name}`);
    }

    return response.json();
}

function initializeMultiSelect() {
    let selectedItems = new Set();
    const batchActions = document.querySelector('.batch-actions');
    const selectedCountSpan = document.querySelector('.selected-count');

    // Handle select all checkbox
    const selectAllCheckbox = document.querySelector('.select-all');
    selectAllCheckbox.addEventListener('change', (e) => {
        const checkboxes = document.querySelectorAll('.document-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = e.target.checked;
            const row = checkbox.closest('tr');
            if (e.target.checked) {
                row.classList.add('selected');
                selectedItems.add(row.dataset.path);
            } else {
                row.classList.remove('selected');
                selectedItems.delete(row.dataset.path);
            }
        });
        updateBatchActionsVisibility();
    });

    // Handle individual checkboxes
    const tableBody = document.querySelector('.documents-table tbody');
    tableBody.addEventListener('click', (e) => {
        const checkbox = e.target.closest('.document-checkbox');
        if (checkbox) {
            const row = checkbox.closest('tr');
            if (checkbox.checked) {
                row.classList.add('selected');
                selectedItems.add(row.dataset.path);
            } else {
                row.classList.remove('selected');
                selectedItems.delete(row.dataset.path);
                selectAllCheckbox.checked = false;
            }
            updateBatchActionsVisibility();
        }
    });

    // Handle Shift+Click for range selection
    let lastChecked = null;
    tableBody.addEventListener('click', (e) => {
        const checkbox = e.target.closest('.document-checkbox');
        if (!checkbox) return;

        if (e.shiftKey && lastChecked) {
            const checkboxes = Array.from(document.querySelectorAll('.document-checkbox'));
            const start = checkboxes.indexOf(checkbox);
            const end = checkboxes.indexOf(lastChecked);
            const [lower, upper] = [Math.min(start, end), Math.max(start, end)];

            checkboxes.slice(lower, upper + 1).forEach(cb => {
                cb.checked = lastChecked.checked;
                const row = cb.closest('tr');
                if (cb.checked) {
                    row.classList.add('selected');
                    selectedItems.add(row.dataset.path);
                } else {
                    row.classList.remove('selected');
                    selectedItems.delete(row.dataset.path);
                }
            });
        }

        lastChecked = checkbox;
        updateBatchActionsVisibility();
    });

    // Batch delete handler
    const batchDeleteButton = document.getElementById('batch-delete-button');
    batchDeleteButton.addEventListener('click', async () => {
        if (selectedItems.size === 0) return;

        if (confirm(`Are you sure you want to delete ${selectedItems.size} selected item(s)?`)) {
            try {
                const deletePromises = Array.from(selectedItems).map(async (path) => {
                    const response = await fetch(`/delete/document?path=${encodeURIComponent(path)}`, {
                        method: 'DELETE'
                    });
                    if (!response.ok) throw new Error(`Failed to delete ${path}`);
                });

                await Promise.all(deletePromises);
                selectedItems.clear(); // Clear the selection
                updateBatchActionsVisibility(); // Hide the delete button
                await loadDocuments(currentFolderPath); // Reload the document list
            } catch (error) {
                console.error('Batch delete error:', error);
                alert('Failed to delete some items');
            }
        }
    });

    function updateBatchActionsVisibility() {
        selectedCountSpan.textContent = selectedItems.size;
        if (selectedItems.size > 0) {
            batchActions.classList.add('visible');
        } else {
            batchActions.classList.remove('visible');
        }
    }
}

function modifyTableRowsForSelection() {
    const rows = document.querySelectorAll('.documents-table tbody tr');
    rows.forEach(row => {
        if (!row.classList.contains('folder-row')) {
            const checkbox = document.createElement('td');
            checkbox.style.width = '30px';
            checkbox.innerHTML = '<input type="checkbox" class="document-checkbox">';
            row.insertBefore(checkbox, row.firstChild);

            // Add path data attribute for deletion
            const docName = row.querySelector('.document-name');
            if (docName) {
                row.dataset.path = docName.dataset.path;
            }
        } else {
            // Add empty cell for folders to maintain alignment
            const emptyCell = document.createElement('td');
            emptyCell.style.width = '30px';
            row.insertBefore(emptyCell, row.firstChild);
        }
    });
}

async function loadDocuments(currentPath = '') {
    currentFolderPath = currentPath;
    try {
        const response = await fetch(`/get/documents?path=${encodeURIComponent(currentPath)}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const tableWrapper = document.querySelector('.documents-table-wrapper');
        if (!tableWrapper) return;

        // Update document count - using total processed documents
        const docCount = document.getElementById('doc-count');
        if (docCount) {
            docCount.textContent = `Total: ${data.total_processed || 0} documents`;
        }

        // Clear the wrapper completely
        tableWrapper.innerHTML = '';

        // Create new table with proper structure
        const table = document.createElement('table');
        table.className = 'documents-table';

        // Create header with checkbox column
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th style="width: 30px;">
                    <input type="checkbox" class="select-checkbox select-all" title="Select all">
                </th>
                <th>Name</th>
                <th>Type</th>
                <th>Size</th>
                <th>Modified</th>
            </tr>
        `;
        table.appendChild(thead);

        // Create tbody
        const tbody = document.createElement('tbody');

        // Add "up" navigation if not in root
        if (currentPath) {
            tbody.innerHTML += `
                <tr class="folder-row" data-path="${encodeURIComponent(getParentPath(currentPath))}">
                    <td style="width: 30px;"></td>
                    <td>
                        <div class="folder-name">
                            <svg class="folder-icon" viewBox="0 0 24 24" width="24" height="24">
                                <path fill="currentColor" d="M3 3l18 0v18H3V3zm2 2v14h14V5H5z"/>
                                <path fill="currentColor" d="M15 11v6h-6v-6h6zm-2-2V7H7v2h6z"/>
                            </svg>
                            ..
                        </div>
                    </td>
                    <td>Folder</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            `;
        }

        // Add folders
        data.folders.forEach(folder => {
            tbody.innerHTML += `
                <tr class="folder-row" data-path="${encodeURIComponent(folder.path)}">
                    <td style="width: 30px;"></td>
                    <td>
                        <div class="folder-name">
                            <svg class="folder-icon" viewBox="0 0 24 24" width="24" height="24">
                                <path fill="currentColor" d="M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2z"/>
                            </svg>
                            ${escapeHtml(folder.name)}
                        </div>
                    </td>
                    <td>Folder</td>
                    <td>-</td>
                    <td>${formatDate(folder.modified)}</td>
                </tr>
            `;
        });

        // Add files
        data.files.forEach(file => {
            const fileName = file.name;
            const lastDotIndex = fileName.lastIndexOf('.');
            let sanitizedName;
            if (lastDotIndex !== -1) {
                const name = fileName.slice(0, lastDotIndex).trim();
                const ext = fileName.slice(lastDotIndex + 1).trim();
                sanitizedName = `${name}.${ext}`;
            } else {
                sanitizedName = fileName.trim();
            }

            tbody.innerHTML += `
                <tr data-path="${encodeURIComponent(file.path)}">
                    <td style="width: 30px;">
                        <input type="checkbox" class="select-checkbox document-checkbox">
                    </td>
                    <td>
                        <span class="document-name" data-path="${encodeURIComponent(file.path)}">${escapeHtml(sanitizedName)}</span>
                    </td>
                    <td>${escapeHtml(file.type)}</td>
                    <td>${formatFileSize(file.size)}</td>
                    <td>${formatDate(file.modified)}</td>
                </tr>
            `;
        });

        table.appendChild(tbody);

        // Add current path display
        const pathDisplay = document.createElement('div');
        pathDisplay.className = 'current-path';
        pathDisplay.innerHTML = `Current Folder: <span class="folder-path">${currentPath ? decodeURIComponent(currentPath) : 'Root'}</span>`;
        tableWrapper.appendChild(pathDisplay);
        tableWrapper.appendChild(table);

        // Initialize multi-select functionality
        initializeMultiSelect();

        // Add click handlers for folder navigation
        const folderRows = tableWrapper.querySelectorAll('.folder-row');
        folderRows.forEach(row => {
            row.addEventListener('click', (e) => {
                if (!e.target.closest('.select-checkbox')) {
                    const path = decodeURIComponent(row.dataset.path)
                        .split('/')
                        .map(segment => encodeURIComponent(segment))
                        .join('/');
                    loadDocuments(path);
                }
            });
        });

        // Add context menu handlers for documents
        const documentNames = tableWrapper.querySelectorAll('.document-name');
        documentNames.forEach(docName => {
            docName.addEventListener('contextmenu', (e) => {
                const filePath = docName.dataset.path;
                const fileName = docName.textContent;
                createContextMenu(e, fileName, filePath);
            });
        });

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

    // Update the loadDocuments function to include new handlers
    const originalLoadDocuments = loadDocuments;
    loadDocuments = async (path = '') => {
        await originalLoadDocuments(path);
        updateFolderContextMenuHandlers();
    };

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

    function sanitizeFileName(filename) {
        const [name, ext] = filename.split(/\.(?=[^.]+$)/);
        return `${name.trim()}.${ext.trim()}`;
    }

    function updateFileCount() {
        const fileCount = fileMap.size;
        const remainingSlots = 10 - fileCount;

        // Update the upload box text to show remaining slots
        const uploadContent = document.querySelector('.upload-content p:first-child');
        if (uploadContent) {
            uploadContent.textContent = `Drag and drop files or click to upload (${fileCount}/10 files)`;
        }
    }

    function handleFiles(files) {
        if (fileMap.size + files.length > 10) {
            alert('You can select no more than 10 files at a time');
            return;
        }

        files.forEach(file => {
            if (isValidFile(file)) {
                const sanitizedName = sanitizeFileName(file.name);

                if (fileMap.has(sanitizedName)) {
                    alert(`File "${sanitizedName}" has already been added`);
                    return;
                }

                const sanitizedFile = new File([file], sanitizedName, {
                    type: file.type,
                    lastModified: file.lastModified
                });

                fileMap.set(sanitizedName, sanitizedFile);
                uploadList.appendChild(createFileItem(sanitizedFile));
                processBtn.style.display = 'block';
            }
        });

        updateFileCount(); // Update the count after adding files
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
            updateFileCount(); // Update the count after removing a file
        });

        return div;
    }

    // Chat Functions

    function formatMessageText(text) {
        // 1. Extract all math segments (inline and display math)
        // This regex will match $...$ or $$...$$ blocks.
        // Note: This is a simplistic regex that assumes the math won't contain unmatched $ signs.
        // It matches either $...$ or $$...$$ and stores them for later.
        const mathSegments = [];
        let mathIndex = 0;
        text = text.replace(/(\${1,2})([\s\S]*?\1)/g, (match) => {
            const placeholder = `<MATH_${mathIndex}>`;
            mathSegments.push(match);
            mathIndex++;
            return placeholder;
        });

        // 2. Apply formatting to the non-math text
        // Make sure none of these replacements affect math placeholders
        text = text.replace(/^# (.+)$/gm, '<h3 class="message-header">$1</h3>');
        text = text.replace(/^([A-Za-z].+)$/gm, '<h4 class="message-subheader">$1</h4>');
        text = text.replace(/(.+?)(?=\n*?• |\n*?$)/g, '<p>$1</p>\n');
        text = text.replace(/• (.+)/g, '<li class="message-bullet">$1</li>');
        text = text.replace(/(<li[^>]*>.*<\/li>\n?)+/g, match => {
            const items = match.split('\n').map(item => `  ${item}`).join('\n');
            return `<ul class="message-list">\n${items}\n</ul>`;
        });
        text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // 3. Restore math segments
        mathSegments.forEach((segment, i) => {
            text = text.replace(`<MATH_${i}>`, segment);
        });

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
            const safeContent = (content.text || content).replace(/</g, "&lt;").replace(/>/g, "&gt;");
            textDiv.innerHTML = safeContent.replace(/\n/g, '<br>');
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

        updateFileCount();
        MathJax.typesetPromise()
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
                formData.append('folder', currentFolderPath); // Add current folder path

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

            // Refresh documents list for current folder
            await loadDocuments(currentFolderPath);

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

    // Add event listeners for action buttons
    const rescanButton = document.getElementById('rescan-button');
    if (rescanButton) {
        rescanButton.addEventListener('click', async () => {
            if (rescanButton.classList.contains('loading')) return;

            rescanButton.classList.add('loading');
            const originalContent = rescanButton.innerHTML;
            rescanButton.innerHTML = `<div class="spinner"></div>Rescanning...`;

            try {
                const response = await fetch('/rescan', { method: 'POST' });
                if (!response.ok) throw new Error('Rescan failed');

                await loadDocuments(currentFolderPath);
            } catch (error) {
                console.error('Rescan error:', error);
                alert('Failed to rescan documents');
            } finally {
                rescanButton.classList.remove('loading');
                rescanButton.innerHTML = originalContent;
            }
        });
    }

    const newFolderButton = document.getElementById('new-folder-button');
    if (newFolderButton) {
        newFolderButton.addEventListener('click', () => {
            const modal = createModal(
                'Create New Folder',
                `<input type="text" class="modal-input" placeholder="Folder name" maxlength="255">`,
                async () => {
                    const input = modal.querySelector('.modal-input');
                    const folderName = input.value.trim();

                    if (!folderName) {
                        alert('Please enter a folder name');
                        return;
                    }

                    try {
                        const response = await fetch('/folder/create', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                parent_path: currentFolderPath,
                                folder_name: folderName
                            })
                        });

                        if (!response.ok) {
                            const error = await response.json();
                            throw new Error(error.detail);
                        }

                        await loadDocuments(currentFolderPath);
                    } catch (error) {
                        console.error('Create folder error:', error);
                        alert(error.message || 'Failed to create folder');
                    }
                }
            );
            // Add focus to input after modal creation
            setTimeout(() => {
                const input = modal.querySelector('.modal-input');
                if (input) input.focus();
            }, 0);
        });
    }

    loadDocuments();
    input.focus();
});