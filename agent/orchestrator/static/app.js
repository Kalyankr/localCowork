/**
 * LocalCowork Web UI Application
 * 
 * Main JavaScript application for the LocalCowork web interface.
 * Handles WebSocket connections, slash commands, message handling,
 * and UI interactions.
 */

// =============================================================================
// State
// =============================================================================

let ws = null;
let sessionId = localStorage.getItem('localcowork_session') || null;
let isProcessing = false;
let currentTaskId = null;
let currentModel = localStorage.getItem('localcowork_model') || 'mistral';
let pendingConfirmId = null;
let startTime = null;
let stepResults = [];

// =============================================================================
// Markdown Configuration
// =============================================================================

if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return typeof hljs !== 'undefined' ? hljs.highlightAuto(code).value : code;
        }
    });
}

// =============================================================================
// Slash Commands
// =============================================================================

const commands = [
    { cmd: '/clear', desc: 'Clear chat history' },
    { cmd: '/status', desc: 'Show connection status' },
    { cmd: '/history', desc: 'Toggle history sidebar' },
    { cmd: '/model', desc: 'Change model (e.g., /model llama3.2)' },
    { cmd: '/help', desc: 'Show available commands' },
];

// =============================================================================
// DOM Elements
// =============================================================================

const chatContainer = document.getElementById('chat-container');
const emptyState = document.getElementById('empty-state');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const statusBadge = document.getElementById('status-badge');
const autocomplete = document.getElementById('autocomplete');
const modelSelect = document.getElementById('model-select');
const sidebar = document.getElementById('sidebar');

// =============================================================================
// WebSocket Connection
// =============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    updateStatus('connecting');
    
    ws.onopen = () => updateStatus('connected');
    ws.onclose = () => {
        updateStatus('disconnected');
        setTimeout(connectWebSocket, 3000);
    };
    ws.onerror = () => updateStatus('disconnected');
    ws.onmessage = (e) => handleWsMessage(JSON.parse(e.data));
}

function updateStatus(status) {
    statusBadge.className = 'status-badge ' + status;
    const label = status.charAt(0).toUpperCase() + status.slice(1);
    statusBadge.innerHTML = `<span class="status-dot"></span>${label}`;
}

function handleWsMessage(msg) {
    console.log('WS:', msg);
    if (msg.type === 'progress' && msg.task_id === currentTaskId) {
        updateProgress(msg);
    } else if (msg.type === 'complete' && msg.task_id === currentTaskId) {
        finishTask(msg);
    } else if (msg.type === 'confirm_request') {
        showConfirmModal(msg);
    }
}

// =============================================================================
// Slash Command Handling
// =============================================================================

function showAutocomplete(input) {
    const filtered = commands.filter(c => c.cmd.startsWith(input.toLowerCase()));
    if (filtered.length === 0) {
        hideAutocomplete();
        return;
    }
    
    autocomplete.innerHTML = filtered.map((c, i) => `
        <div class="autocomplete-item ${i === 0 ? 'selected' : ''}" onclick="selectCommand('${c.cmd}')">
            <span class="cmd">${c.cmd}</span>
            <span class="desc">${c.desc}</span>
        </div>
    `).join('');
    autocomplete.classList.add('show');
}

function hideAutocomplete() {
    autocomplete.classList.remove('show');
}

function selectCommand(cmd) {
    messageInput.value = cmd + ' ';
    hideAutocomplete();
    messageInput.focus();
}

function executeCommand(input) {
    const parts = input.trim().split(' ');
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1).join(' ');
    
    switch (cmd) {
        case '/clear':
            clearChat();
            return true;
        case '/status':
            showStatus();
            return true;
        case '/history':
            toggleSidebar();
            return true;
        case '/model':
            if (args) {
                changeModel(args);
                addMessage(`Model changed to: ${args}`, 'system');
            } else {
                addMessage(`Current model: ${currentModel}. Usage: /model <name>`, 'system');
            }
            return true;
        case '/help':
            showHelp();
            return true;
    }
    return false;
}

function showHelp() {
    const helpText = commands.map(c => `${c.cmd} - ${c.desc}`).join('\n');
    addMessage(`Available commands:\n${helpText}`, 'system');
}

function showStatus() {
    const status = `Status Information:
• Connection: ${statusBadge.textContent}
• Model: ${currentModel}
• Session: ${sessionId ? sessionId.slice(0, 8) + '...' : 'None'}
• Version: v0.3.0`;
    addMessage(status, 'system');
}

function changeModel(model) {
    currentModel = model;
    modelSelect.value = model;
    localStorage.setItem('localcowork_model', model);
}

function toggleSidebar() {
    sidebar.classList.toggle('hidden');
    document.getElementById('sidebar-toggle').classList.toggle('active');
}

// =============================================================================
// Message Handling
// =============================================================================

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isProcessing) return;
    
    // Check for slash commands
    if (text.startsWith('/')) {
        if (executeCommand(text)) {
            messageInput.value = '';
            hideAutocomplete();
            return;
        }
    }
    
    hideAutocomplete();
    if (emptyState) emptyState.style.display = 'none';
    
    addMessage(text, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    isProcessing = true;
    sendBtn.disabled = true;
    startTime = Date.now();
    stepResults = [];
    
    const progressEl = showProgress();
    
    try {
        const response = await fetch('/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                request: text,
                session_id: sessionId,
            }),
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Request failed');
        }
        
        const data = await response.json();
        sessionId = data.session_id;
        localStorage.setItem('localcowork_session', sessionId);
        currentTaskId = data.task_id;
        
        // Subscribe to updates
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'subscribe', task_id: data.task_id }));
        }
        
        progressEl.remove();
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        
        if (data.response) {
            addMessage(data.response, 'assistant', { 
                duration, 
                steps: data.steps 
            });
        } else if (data.status === 'failed') {
            addMessage('Sorry, I encountered an error.', 'error');
        }
        
        // Save to history
        saveToHistory(text, data.response);
        
    } catch (err) {
        progressEl.remove();
        addMessage('Error: ' + err.message, 'error');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

function addMessage(text, type, meta = null) {
    const msg = document.createElement('div');
    msg.className = 'message ' + type;
    
    if (type === 'assistant' && typeof marked !== 'undefined') {
        // Render markdown for assistant messages
        const mdDiv = document.createElement('div');
        mdDiv.className = 'md-content';
        mdDiv.innerHTML = marked.parse(text);
        msg.appendChild(mdDiv);
        
        // Add copy buttons to code blocks
        msg.querySelectorAll('pre').forEach(pre => {
            const copyBtn = document.createElement('button');
            copyBtn.className = 'code-copy-btn';
            copyBtn.textContent = 'Copy';
            copyBtn.onclick = () => {
                const code = pre.querySelector('code')?.textContent || pre.textContent;
                navigator.clipboard.writeText(code).then(() => {
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => copyBtn.textContent = 'Copy', 1500);
                });
            };
            pre.style.position = 'relative';
            pre.appendChild(copyBtn);
        });
        
        // Rewrite image src to use /files/ endpoint for local paths
        msg.querySelectorAll('img').forEach(img => {
            const src = img.getAttribute('src') || '';
            if (src && !src.startsWith('http') && !src.startsWith('/files/') && !src.startsWith('data:')) {
                img.setAttribute('src', '/files/' + src.replace(/^\/+/, ''));
            }
            img.classList.add('chat-image');
            img.addEventListener('click', () => openImageModal(img.src));
        });
        
        // Detect image file paths mentioned in text and render inline
        renderInlineImages(mdDiv);
    } else {
        msg.textContent = text;
    }
    
    // Add metadata (duration, steps)
    if (meta && type === 'assistant') {
        const metaEl = document.createElement('div');
        metaEl.className = 'message-meta';
        if (meta.duration) {
            metaEl.innerHTML += `<span>\u2713 ${meta.duration}s</span>`;
        }
        if (meta.steps) {
            metaEl.innerHTML += `<span>\u2022 ${meta.steps} steps</span>`;
        }
        msg.appendChild(metaEl);
    }
    
    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    msg.appendChild(timestamp);
    
    // Add copy button for assistant messages
    if (type === 'assistant') {
        const actions = document.createElement('div');
        actions.className = 'message-actions';
        const copyMsgBtn = document.createElement('button');
        copyMsgBtn.className = 'msg-action-btn';
        copyMsgBtn.textContent = '\ud83d\udccb Copy';
        copyMsgBtn.onclick = () => {
            navigator.clipboard.writeText(text).then(() => {
                copyMsgBtn.textContent = '\u2713 Copied';
                setTimeout(() => copyMsgBtn.textContent = '\ud83d\udccb Copy', 1500);
            });
        };
        actions.appendChild(copyMsgBtn);
        msg.appendChild(actions);
    }
    
    chatContainer.appendChild(msg);
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
}

// =============================================================================
// Progress Display
// =============================================================================

function showProgress() {
    const el = document.createElement('div');
    el.className = 'progress-container';
    el.id = 'progress-indicator';
    el.innerHTML = `
        <div class="progress-spinner"></div>
        <div class="progress-content">
            <div class="progress-action">Thinking...</div>
            <div class="progress-steps"></div>
        </div>
    `;
    chatContainer.appendChild(el);
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    return el;
}

function updateProgress(msg) {
    const el = document.getElementById('progress-indicator');
    if (!el) return;
    
    const actionEl = el.querySelector('.progress-action');
    const stepsEl = el.querySelector('.progress-steps');
    
    // Handle parallel sub-agent progress
    if (msg.status === 'parallel') {
        el.classList.add('parallel-mode');
        actionEl.innerHTML = `<span class="parallel-icon">⚡</span> ${msg.thought || 'Running subtasks in parallel'}`;
        actionEl.className = 'progress-action parallel';
        
        // Parse subtask descriptions from action field
        if (msg.action) {
            const subtasks = msg.action.split(', ').map(s => s.trim());
            stepsEl.innerHTML = `
                <div class="subtask-list">
                    ${subtasks.map((s, i) => `
                        <div class="subtask-item" id="subtask-${i}">
                            <span class="subtask-spinner"></span>
                            <span class="subtask-text">${s}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
        return;
    }
    
    // Update action text
    let actionText = 'Thinking...';
    let actionClass = '';
    
    if (msg.action) {
        if (msg.action.startsWith('shell:')) {
            actionText = '$ ' + msg.action.slice(6).trim().slice(0, 50);
            actionClass = 'shell';
        } else if (msg.action.startsWith('python:')) {
            actionText = 'Running Python...';
            actionClass = 'python';
        } else if (msg.action.startsWith('web_search:')) {
            actionText = 'Searching the web...';
            actionClass = 'web';
        } else if (msg.action.startsWith('fetch_webpage:')) {
            actionText = 'Fetching webpage...';
            actionClass = 'web';
        }
    }
    
    actionEl.textContent = actionText;
    actionEl.className = 'progress-action ' + actionClass;
    
    // Update step dots
    if (msg.status === 'success' || msg.status === 'error') {
        stepResults.push(msg.status);
    }
    
    // Handle completed/partial status for parallel tasks
    if (msg.status === 'completed' || msg.status === 'partial') {
        const subtaskList = el.querySelector('.subtask-list');
        if (subtaskList) {
            // Update subtask items to show completion
            const items = subtaskList.querySelectorAll('.subtask-item');
            items.forEach(item => {
                item.classList.add('completed');
                const spinner = item.querySelector('.subtask-spinner');
                if (spinner) spinner.outerHTML = '<span class="subtask-check">✓</span>';
            });
        }
    }
    
    let stepsHtml = stepResults.slice(-8).map(s => 
        `<div class="step-dot ${s}"></div>`
    ).join('');
    
    if (stepResults.length > 0) {
        stepsHtml += `<span class="progress-iteration">Step ${msg.iteration || stepResults.length}</span>`;
    }
    
    stepsEl.innerHTML = stepsHtml;
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
}

function finishTask(msg) {
    const el = document.getElementById('progress-indicator');
    if (el) el.remove();
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    if (msg.response) {
        addMessage(msg.response, 'assistant', { 
            duration, 
            steps: stepResults.length 
        });
    }
    
    isProcessing = false;
    sendBtn.disabled = false;
}

// =============================================================================
// Confirmation Modal
// =============================================================================

function showConfirmModal(msg) {
    pendingConfirmId = msg.confirm_id;
    document.getElementById('confirm-message').textContent = msg.message || msg.reason || 'This action requires confirmation.';
    document.getElementById('confirm-command').textContent = msg.command || '';
    document.getElementById('confirm-modal').classList.remove('hidden');
}

function handleConfirm(confirmed) {
    document.getElementById('confirm-modal').classList.add('hidden');
    
    // Show user's decision clearly
    const decisionMessage = confirmed 
        ? '✓ Approved - Continuing with operation...'
        : '✗ Denied - Operation cancelled';
    const decisionClass = confirmed ? 'success' : 'error';
    
    // Add decision to current message steps
    if (currentMessageEl) {
        const stepsContainer = currentMessageEl.querySelector('.agent-steps');
        if (stepsContainer) {
            const decisionStep = document.createElement('div');
            decisionStep.className = `step ${decisionClass}`;
            decisionStep.innerHTML = `
                <div class="step-header">
                    <span class="step-icon">${confirmed ? '✓' : '✗'}</span>
                    <span class="step-title">User Decision</span>
                </div>
                <div class="step-content">${confirmed ? 'Permission granted - proceeding' : 'Permission denied - operation cancelled'}</div>
            `;
            stepsContainer.appendChild(decisionStep);
        }
    }
    
    // Show toast notification
    showToast(decisionMessage, decisionClass);
    
    if (ws && ws.readyState === WebSocket.OPEN && pendingConfirmId) {
        ws.send(JSON.stringify({
            type: 'confirm_response',
            confirm_id: pendingConfirmId,
            confirmed: confirmed,
        }));
    }
    pendingConfirmId = null;
}

function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10000;';
        document.body.appendChild(toastContainer);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    toastContainer.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// =============================================================================
// Chat Management
// =============================================================================

function clearChat() {
    chatContainer.innerHTML = '';
    const newEmpty = document.createElement('div');
    newEmpty.className = 'empty-state';
    newEmpty.id = 'empty-state';
    newEmpty.innerHTML = `
        <div class="empty-logo-mark">LC</div>
        <h2>What can I help you with?</h2>
        <p>Run commands, write code, analyze files, search the web \u2014 powered by your local LLM.</p>
        <div class="examples">
            <button class="example-btn" onclick="useExample(this)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 7h.01M11 7h6"/></svg>
                <span class="example-text">List files in current directory</span>
            </button>
            <button class="example-btn" onclick="useExample(this)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <span class="example-text">Find Python files over 1MB</span>
            </button>
            <button class="example-btn" onclick="useExample(this)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                <span class="example-text">Check disk usage breakdown</span>
            </button>
            <button class="example-btn" onclick="useExample(this)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                <span class="example-text">Show system information</span>
            </button>
        </div>
        <div class="empty-hint"><kbd>Enter</kbd> to send &middot; <kbd>Shift+Enter</kbd> new line &middot; <kbd>/</kbd> for commands</div>
    `;
    chatContainer.appendChild(newEmpty);
    sessionId = null;
    localStorage.removeItem('localcowork_session');
}

function useExample(btn) {
    const textEl = btn.querySelector('.example-text');
    messageInput.value = textEl ? textEl.textContent : btn.textContent;
    messageInput.focus();
}

// =============================================================================
// Image Display
// =============================================================================

const IMAGE_EXTENSIONS = /\.(png|jpg|jpeg|gif|webp|svg|bmp)$/i;
const IMAGE_PATH_REGEX = /(?:^|\s)((?:\/|\.\/|\.\.\/)?[\w./_-]+\.(?:png|jpg|jpeg|gif|webp|svg|bmp))\b/gi;

function renderInlineImages(container) {
    // Walk text nodes looking for image file paths not already inside <img> or <a>
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const matches = [];
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.parentElement?.closest('code, pre, a, img')) continue;
        let m;
        IMAGE_PATH_REGEX.lastIndex = 0;
        while ((m = IMAGE_PATH_REGEX.exec(node.textContent)) !== null) {
            matches.push({ node, path: m[1].trim() });
        }
    }
    // Deduplicate paths already shown as <img>
    const existingSrcs = new Set([...container.querySelectorAll('img')].map(i => i.src));
    for (const { path } of matches) {
        const src = '/files/' + path.replace(/^\/+/, '');
        if (existingSrcs.has(window.location.origin + src)) continue;
        existingSrcs.add(window.location.origin + src);
        const wrapper = document.createElement('div');
        wrapper.className = 'inline-image-wrapper';
        const img = document.createElement('img');
        img.src = src;
        img.alt = path;
        img.className = 'chat-image';
        img.addEventListener('click', () => openImageModal(img.src));
        img.addEventListener('error', () => wrapper.remove());
        wrapper.appendChild(img);
        const caption = document.createElement('div');
        caption.className = 'image-caption';
        caption.textContent = path;
        wrapper.appendChild(caption);
        container.appendChild(wrapper);
    }
}

function openImageModal(src) {
    let overlay = document.getElementById('image-modal');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'image-modal';
        overlay.className = 'image-modal-overlay';
        overlay.addEventListener('click', () => overlay.classList.add('hidden'));
        overlay.innerHTML = '<img class="image-modal-img" />';
        document.body.appendChild(overlay);
    }
    overlay.querySelector('img').src = src;
    overlay.classList.remove('hidden');
}

// =============================================================================
// History Management
// =============================================================================

function saveToHistory(request, response) {
    const history = JSON.parse(localStorage.getItem('localcowork_history') || '[]');
    history.unshift({
        id: Date.now(),
        request: request.slice(0, 100),
        response: response?.slice(0, 100),
        timestamp: new Date().toISOString(),
    });
    localStorage.setItem('localcowork_history', JSON.stringify(history.slice(0, 20)));
    renderHistory();
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('localcowork_history') || '[]');
    const container = document.getElementById('history-list');
    
    if (history.length === 0) {
        container.innerHTML = '<div class="no-history">No conversation history yet</div>';
        return;
    }
    
    container.innerHTML = history.map(h => `
        <div class="history-item" onclick="loadHistoryItem('${h.request.replace(/'/g, "\\'")}')">
            <div class="history-item-title">${h.request}</div>
            <div class="history-item-meta">${new Date(h.timestamp).toLocaleDateString()}</div>
        </div>
    `).join('');
}

function loadHistoryItem(request) {
    messageInput.value = request;
    messageInput.focus();
    if (window.innerWidth < 768) toggleSidebar();
}

// =============================================================================
// Event Listeners
// =============================================================================

// Input autocomplete
messageInput.addEventListener('input', (e) => {
    const value = e.target.value;
    if (value.startsWith('/')) {
        showAutocomplete(value);
    } else {
        hideAutocomplete();
    }
    // Auto-resize textarea
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+K - Clear
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        clearChat();
    }
    // Ctrl+H - History
    if (e.ctrlKey && e.key === 'h') {
        e.preventDefault();
        toggleSidebar();
    }
    // Ctrl+I - Status
    if (e.ctrlKey && e.key === 'i') {
        e.preventDefault();
        showStatus();
    }
    // Escape - Close modals/autocomplete
    if (e.key === 'Escape') {
        hideAutocomplete();
        document.getElementById('confirm-modal').classList.add('hidden');
    }
    // Enter to send (Shift+Enter for new line in textarea)
    if (e.key === 'Enter' && document.activeElement === messageInput && !e.shiftKey && !isProcessing) {
        e.preventDefault();
        sendMessage();
    }
    // Tab to select autocomplete
    if (e.key === 'Tab' && autocomplete.classList.contains('show')) {
        e.preventDefault();
        const selected = autocomplete.querySelector('.autocomplete-item.selected');
        if (selected) {
            selectCommand(selected.querySelector('.cmd').textContent);
        }
    }
});

// =============================================================================
// Initialize
// =============================================================================

function init() {
    // Set model selector to current model
    modelSelect.value = currentModel;
    
    // Connect WebSocket
    connectWebSocket();
    
    // Render history
    renderHistory();
    
    // Focus input
    messageInput.focus();
}

// Run initialization when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// If DOM is already loaded, initialize immediately
if (document.readyState !== 'loading') {
    init();
}
