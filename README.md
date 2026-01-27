# 🤖 LocalCowork

**A privacy-first AI agent that runs entirely on your machine.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com/)

---

LocalCowork is a **pure agentic** AI assistant inspired by Claude's coding agent. It uses shell commands and Python to accomplish tasks—adapting step-by-step based on what it discovers. **100% local, 100% private.**

```bash
$ localcowork

🤖 LocalCowork — Pure Agentic AI
   Model: mistral | Type 'exit' to quit

You: organize my downloads by file type

  🤔 I'll first see what's in your Downloads folder
  ⚡ shell: ls -la ~/Downloads

  🤔 I see images, PDFs, and documents. Let me organize them.
  ⚡ shell: mkdir -p ~/Downloads/{Images,Documents,PDFs}
  ⚡ shell: mv ~/Downloads/*.jpg ~/Downloads/*.png ~/Downloads/Images/
  ⚡ shell: mv ~/Downloads/*.pdf ~/Downloads/PDFs/
  ⚡ shell: mv ~/Downloads/*.doc* ~/Downloads/*.txt ~/Downloads/Documents/

  ✓ Done! Organized your downloads into 3 folders:
    - Images/ (12 files)
    - Documents/ (5 files)  
    - PDFs/ (8 files)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **Privacy First** | All processing happens locally—your files never leave your machine |
| 🤖 **Pure Agentic** | ReAct loop: Observe → Think → Act → Repeat |
| 🐚 **Shell + Python** | Uses tools you already know—no complex APIs to learn |
| 💬 **Conversational** | Remembers context across your session |
| 🐳 **Sandboxed Python** | Code runs in isolated Docker containers |
| 🌐 **Web UI & API** | REST API + WebSocket for real-time streaming |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker** (for sandboxed Python execution)
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
uv run localcowork
```

---

## 📖 Usage

### Interactive Mode (Recommended)

```bash
# Start the agent
localcowork

# You can have a conversation
You: hello
Agent: Hi! I'm LocalCowork. I can help with files, web, and automation.

You: what's in my downloads?
Agent: [runs ls ~/Downloads and shows results]

You: move the PDFs to a new folder called Reports
Agent: [creates folder, moves files, confirms]
```

### CLI Options

```bash
localcowork              # Interactive mode (default)
localcowork --model llama3   # Use a different model
localcowork serve        # Start the web API server
```

### Web API

```bash
# Start the API server
localcowork serve

# POST to /run
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"request": "list files in my home directory"}'
```

---

## 🧠 How It Works

LocalCowork uses a **ReAct (Reasoning + Acting)** loop:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Request   │────▶│   Think     │────▶│    Act      │
│             │     │  (Ollama)   │     │ (shell/py)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           ▲                   │
                           └───────────────────┘
                              Observe result
```

### The Agent Has Two Tools

| Tool | What It Does |
|------|--------------|
| **shell** | Run any bash command (`ls`, `mv`, `grep`, `curl`, etc.) |
| **python** | Execute Python code for complex logic |

That's it. No complex tool schemas. The LLM already knows bash and Python.

### Example Agent Reasoning

```
Request: "Find large files over 100MB"

Step 1:
  Thought: I'll use find to search for large files
  Action: shell → find ~ -size +100M -type f 2>/dev/null

Step 2:
  Thought: Found 3 files. Let me format this nicely for the user.
  Action: complete → "Found 3 files over 100MB: ..."
```

---

## ⚙️ Configuration

Environment variables (prefix: `LOCALCOWORK_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALCOWORK_OLLAMA_MODEL` | `mistral` | LLM model to use |
| `LOCALCOWORK_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `LOCALCOWORK_SANDBOX_TIMEOUT` | `30` | Python execution timeout (seconds) |

```bash
# Use a different model
LOCALCOWORK_OLLAMA_MODEL=llama3 localcowork
```

---

## 🔒 Security

- **Path Traversal Protection**: Blocks `../../etc/passwd` style attacks
- **Sensitive Path Blocking**: Denies access to `/etc/shadow`, `~/.ssh`, etc.
- **Hardened Docker Sandbox** (for Python):
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
│   │   ├── __init__.py        # CLI entry point
│   │   └── agent_loop.py      # Interactive agent loop
│   ├── config.py              # Centralized settings
│   ├── llm/
│   │   ├── client.py          # Ollama client
│   │   └── prompts.py         # ReAct prompts
│   ├── orchestrator/
│   │   ├── react_agent.py     # The ReAct agent (core)
│   │   ├── server.py          # FastAPI server
│   │   └── deps.py            # Dependency injection
│   ├── sandbox/
│   │   └── sandbox_runner.py  # Docker sandbox for Python
│   ├── security.py            # Path validation
│   └── tools/                 # Fallback tool implementations
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

# Format code
uv run ruff format .
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built for local-first AI</b>
</p>
