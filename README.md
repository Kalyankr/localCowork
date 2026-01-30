# 🤖 LocalCowork

**A privacy-first AI agent that runs entirely on your machine - inspired by Claude's Cowork.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com/)

---

LocalCowork is a **pure agentic** AI assistant that uses shell commands, Python, and web search to accomplish tasks-adapting step-by-step based on what it discovers. **100% local, 100% private.**

```
  ██╗  LocalCowork v0.3.0  ● mistral
  ███████╗  Type a request or /help
  ──────────────────────────────────────

  ╭──────────────────────────────────────╮
  │ > organize my downloads by file type │
  ╰──────────────────────────────────────╯

  ◐ $ mkdir -p ~/Downloads/{Images,PDFs,Documents}  ●●●

  ◆ LocalCowork
    Done! Organized your downloads into 3 folders:
    - Images/ (12 files)
    - Documents/ (5 files)
    - PDFs/ (8 files)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **Privacy First** | All processing happens locally-your files never leave your machine |
| 🤖 **Pure Agentic** | ReAct loop: Observe → Think → Act → Repeat |
| 🐚 **Shell + Python** | Uses tools you already know—no complex APIs |
| 🌐 **Web Search** | Search the web and fetch webpage content |
| 💬 **Conversational** | Remembers context across your session |
| 🐳 **Sandboxed Python** | Code runs in isolated Docker containers |
| ⏱️ **Long-running Tasks** | 10 min shell / 5 min Python timeouts |
| 🎨 **Professional CLI** | Clean interface with progress indicators |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker** (optional, for sandboxed Python execution)
- **Ollama** with a model installed

### Installation

```bash
# 1. Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# 2. Clone and install
git clone https://github.com/Kalyankr/localCowork.git
cd localCowork
uv sync  # or: pip install -e .

# 3. Start the agent
localcowork
```

---

## 📖 Usage

### Interactive Mode

```bash
localcowork                    # Start interactive mode
localcowork --model llama3     # Use a different model
localcowork serve              # Start the web API server
```

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help and examples |
| `/clear` | Clear screen and history |
| `/status` | Show current settings |
| `/history` | Show conversation history |
| `/model <name>` | Switch to a different model |
| `/quit` | Exit the agent |

### Example Conversations

```
You: analyze sales.csv and show top customers

  ◐ Running Python...  ●

  ◆ LocalCowork
    📊 Sales Analysis (1,247 records):
    - Total revenue: $2.4M
    - Top customer: Acme Corp ($89,400)
    - Growth: +38% YoY
```

```
You: search for python asyncio best practices

  ◐ Searching the web...  ●

  ◆ LocalCowork
    Here are some asyncio best practices:
    1. Use async/await consistently - https://...
    2. Avoid blocking calls in async code - https://...
    3. Use asyncio.gather for concurrent tasks

You: what's the weather API endpoint for Seattle?

  ◐ $ curl wttr.in/Seattle?format=3  ●

  ◆ LocalCowork
    Seattle: ⛅ +12°C
```

---

## 🧠 How It Works

LocalCowork uses a **ReAct (Reasoning + Acting)** loop:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Request   │────▶│   Think     │────▶│    Act      │
│             │     │  (Ollama)   │     │ (tools)     │
└─────────────┘     └─────────────┘     └─────────────┘
                          ▲                    │
                          └────────────────────┘
                              Observe result
```

### Available Tools

| Tool | What It Does |
|------|--------------|
| **shell** | Run any bash command (`ls`, `mv`, `grep`, `curl`, etc.) |
| **python** | Execute Python code (pandas, requests, openpyxl available) |
| **web_search** | Search the web via DuckDuckGo |
| **fetch_webpage** | Fetch and extract text from URLs |

### Example Agent Reasoning

```
Request: "Search for Python 3.13 new features and summarize"

Step 1:
  Thought: I'll search the web for Python 3.13 features
  Action: web_search → {"query": "Python 3.13 new features"}

Step 2:
  Thought: Found good results. Let me fetch the official docs.
  Action: fetch_webpage → {"url": "https://docs.python.org/3.13/whatsnew"}

Step 3:
  Thought: Got the content. I can now summarize for the user.
  Action: complete → "Python 3.13 brings: 1. Improved error messages..."
```

---

## ⚙️ Configuration

Environment variables (prefix: `LOCALCOWORK_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALCOWORK_OLLAMA_MODEL` | `mistral` | LLM model to use |
| `LOCALCOWORK_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `LOCALCOWORK_SHELL_TIMEOUT` | `600` | Shell command timeout (seconds) |
| `LOCALCOWORK_SANDBOX_TIMEOUT` | `300` | Python execution timeout (seconds) |
| `LOCALCOWORK_MAX_AGENT_ITERATIONS` | `15` | Max ReAct loop iterations |

```bash
# Use a different model with longer timeout
LOCALCOWORK_OLLAMA_MODEL=llama3 LOCALCOWORK_SHELL_TIMEOUT=900 localcowork
```

---

## 🔒 Security

### Safety Confirmations
Dangerous operations (file deletion, system changes) require explicit user confirmation before execution.

### Path Protection
- **Path Traversal Protection**: Blocks `../../etc/passwd` style attacks
- **Sensitive Path Blocking**: Denies access to `/etc/shadow`, `~/.ssh`, etc.

### Sandboxed Python (Docker mode)
- No network access (`--network none`)
- Non-root user (`--user 1000:1000`)
- All capabilities dropped (`--cap-drop ALL`)
- Read-only filesystem (`--read-only`)
- Limited resources (256MB RAM, 1 CPU, 50 PIDs)

---

## 📁 Project Structure

```
localCowork/
├── agent/
│   ├── cli/
│   │   ├── __init__.py        # CLI entry point (Typer)
│   │   ├── agent_loop.py      # Interactive agent loop
│   │   └── console.py         # Rich console utilities
│   ├── config.py              # Centralized settings (Pydantic)
│   ├── llm/
│   │   ├── client.py          # Ollama client
│   │   └── prompts.py         # ReAct prompts
│   ├── orchestrator/
│   │   ├── react_agent.py     # The ReAct agent (core)
│   │   ├── agent_models.py    # Agent state models
│   │   ├── server.py          # FastAPI server
│   │   └── deps.py            # Dependency injection
│   ├── sandbox/
│   │   └── sandbox_runner.py  # Docker/permissive sandbox
│   ├── web.py                 # Web search & fetch tools
│   ├── safety.py              # Command/code analysis
│   └── security.py            # Path validation
├── pyproject.toml
└── README.md
```

---

## 🤝 Contributing

Contributions welcome!

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format & lint
uv run ruff format .
uv run ruff check --fix .
```

---

<p align="center">
  <b>Built for local-first AI — Inspired by Claude's Cowork</b>
</p>
