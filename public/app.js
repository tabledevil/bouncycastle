console.log('app.js loaded');

const dropZone = document.getElementById('drop-zone');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const uploadContainer = document.getElementById('upload-container');
const treeContainer = document.getElementById('tree-container');
const xmlTree = document.getElementById('xml-tree');

// --- File Picker ---
uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// --- Drag and Drop ---
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
});

function handleFile(file) {
    console.log('File uploaded:', file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
        const xmlString = e.target.result;
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlString, "text/xml");

        // Check for parsing errors
        const parseError = xmlDoc.querySelector('parsererror');
        if (parseError) {
            console.error('Error parsing XML:', parseError);
            alert('Error parsing XML file. Please check the console for details.');
            return;
        }

        createTreeView(xmlDoc);
        uploadContainer.style.display = 'none';
        treeContainer.style.display = 'block';
    };
    reader.readAsText(file);
}

function createTreeView(xmlDoc) {
    xmlTree.innerHTML = ''; // Clear previous tree
    const rootElement = xmlDoc.documentElement;
    const treeNode = buildTreeNode(rootElement);
    xmlTree.appendChild(treeNode);
}

function buildTreeNode(element) {
    const node = document.createElement('div');
    node.className = 'tree-node';

    const nodeContent = document.createElement('span');
    nodeContent.className = 'node-content';
    nodeContent.textContent = `<${element.nodeName}>`;
    node.appendChild(nodeContent);

    const childrenContainer = document.createElement('div');
    childrenContainer.className = 'children-container';
    node.appendChild(childrenContainer);

    // Recursively build tree for child nodes
    for (const child of element.childNodes) {
        if (child.nodeType === Node.TEXT_NODE && child.textContent.trim() === '') {
            continue; // Skip empty text nodes
        }
        if (child.nodeType === Node.ELEMENT_NODE) {
            childrenContainer.appendChild(buildTreeNode(child));
        } else if (child.nodeType === Node.TEXT_NODE) {
            const textNode = document.createElement('div');
            textNode.className = 'text-node';
            textNode.textContent = child.textContent.trim();
            childrenContainer.appendChild(textNode);
        }
    }

    return node;
}
