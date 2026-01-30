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
    statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
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
    msg.textContent = text;
    
    // Add metadata (duration, steps)
    if (meta && type === 'assistant') {
        const metaEl = document.createElement('div');
        metaEl.className = 'message-meta';
        if (meta.duration) {
            metaEl.innerHTML += `<span>✓ ${meta.duration}s</span>`;
        }
        if (meta.steps) {
            metaEl.innerHTML += `<span>• ${meta.steps} steps</span>`;
        }
        msg.appendChild(metaEl);
    }
    
    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
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
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return el;
}

function updateProgress(msg) {
    const el = document.getElementById('progress-indicator');
    if (!el) return;
    
    const actionEl = el.querySelector('.progress-action');
    const stepsEl = el.querySelector('.progress-steps');
    
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
    
    let stepsHtml = stepResults.slice(-8).map(s => 
        `<div class="step-dot ${s}"></div>`
    ).join('');
    
    if (stepResults.length > 0) {
        stepsHtml += `<span class="progress-iteration">Step ${msg.iteration || stepResults.length}</span>`;
    }
    
    stepsEl.innerHTML = stepsHtml;
    chatContainer.scrollTop = chatContainer.scrollHeight;
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
    
    if (ws && ws.readyState === WebSocket.OPEN && pendingConfirmId) {
        ws.send(JSON.stringify({
            type: 'confirm_response',
            confirm_id: pendingConfirmId,
            confirmed: confirmed,
        }));
    }
    pendingConfirmId = null;
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
        <h2>👋 Hey there!</h2>
        <p>I'm your local AI assistant. Ask me anything or give me a task.</p>
        <div class="examples">
            <button class="example-btn" onclick="useExample(this)">List files in home directory</button>
            <button class="example-btn" onclick="useExample(this)">What's the current date?</button>
            <button class="example-btn" onclick="useExample(this)">Search for Python files</button>
            <button class="example-btn" onclick="useExample(this)">Check disk usage</button>
        </div>
    `;
    chatContainer.appendChild(newEmpty);
    sessionId = null;
    localStorage.removeItem('localcowork_session');
}

function useExample(btn) {
    messageInput.value = btn.textContent;
    messageInput.focus();
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
    // Enter to send
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
