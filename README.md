# 🤖 LocalCowork

**A privacy-first AI assistant that runs entirely on your machine.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com/)

---

LocalCowork transforms natural language requests into executable multi-step plans. Inspired by Claude's agentic capabilities, it brings AI automation to your terminal—**100% locally, 100% private**.

```bash
$ localcowork run "Organize my downloads by file type"

📋 Plan — 4 steps (3× file_op, 1× python)

Execute this plan? [Y/n]: y

✓ list_files     done    List files in ~/Downloads
✓ categorize     done    Group by extension  
✓ move_images    done    Move to Images/
✓ move_pdfs      done    Move to Documents/

✓ 4 succeeded, 0 failed
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **Privacy First** | All processing happens locally—your files never leave your machine |
| 🤖 **Agentic Mode** | ReAct loop with step-by-step reasoning and dynamic adaptation |
| ⚡ **Parallel Execution** | Independent steps run concurrently for faster results |
| 🔍 **Plan Approval** | Review exactly what will happen before execution |
| 🐳 **Hardened Sandbox** | Python runs in isolated Docker containers (no network, no root) |
| 🌐 **Web UI & API** | REST API + WebSocket for real-time task streaming |
| 📊 **Task History** | Persistent history of all tasks with workspace isolation |

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

# 3. Run your first task
uv run localcowork run "List all PDF files in my downloads"
```

---

## 📖 Usage

### CLI Commands

```bash
# Run a task (with plan approval)
localcowork run "organize my downloads folder"

# Skip confirmation
localcowork run "move images to Pictures" --yes

# Agentic mode (step-by-step reasoning)
localcowork run "find large files and summarize" --agentic

# Preview plan without executing
localcowork run "delete temp files" --dry-run

# Verbose output
localcowork run "process documents" --verbose
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--yes` | `-y` | Skip confirmation prompt |
| `--agentic` | `-a` | Use ReAct agent (step-by-step reasoning) |
| `--dry-run` | `-n` | Show plan without executing |
| `--no-parallel` | `-s` | Run steps sequentially |
| `--verbose` | `-v` | Enable debug output |
| `--json` | | Output results as JSON |

### Web Server

```bash
# Start the API server with Web UI
localcowork serve

# Open http://localhost:8000 in your browser
```

### Task Management

```bash
# List recent tasks
localcowork tasks

# Show task details
localcowork task <task-id>

# Cancel a running task
localcowork cancel <task-id>
```

---

## 🛠️ Available Tools

| Tool | Operations |
|------|------------|
| **file_op** | list, read, write, move, copy, delete, mkdir, rename, find |
| **web_op** | fetch, search, download, check |
| **pdf_op** | extract text/metadata, merge, split, page count |
| **data_op** | csv↔excel↔json conversion, preview, stats, filter |
| **archive_op** | zip, unzip, tar, extract (with zip-slip protection) |
| **json_op** | read, write, query, filter, merge, flatten, diff |
| **text_op** | summarize, extract, transform |
| **shell_op** | run safe commands, sysinfo |
| **python** | sandboxed code execution |

---

## ⚙️ Configuration

Environment variables (prefix: `LOCALCOWORK_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALCOWORK_OLLAMA_MODEL` | `mistral` | LLM model to use |
| `LOCALCOWORK_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `LOCALCOWORK_SANDBOX_TIMEOUT` | `30` | Code execution timeout (seconds) |
| `LOCALCOWORK_REQUIRE_APPROVAL` | `true` | Require plan approval |

```bash
# Use a different model
LOCALCOWORK_OLLAMA_MODEL=llama3 localcowork run "summarize notes"
```

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Request   │────▶│   Planner   │────▶│  Executor   │
│             │     │  (Ollama)   │     │  (Parallel) │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             ┌────────────┐            ┌────────────┐             ┌────────────┐
             │   Tools    │            │  Sandbox   │             │  ReAct     │
             │  Registry  │            │  (Docker)  │             │  Agent     │
             └────────────┘            └────────────┘             └────────────┘
```

**Classic Mode**: Plan all steps upfront → Execute in parallel  
**Agentic Mode**: Observe → Reason → Act → Repeat (with reflection)

---

## 🔒 Security

- **Path Traversal Protection**: Blocks `../../etc/passwd` style attacks
- **Sensitive Path Blocking**: Denies access to `/etc/shadow`, `~/.ssh`, etc.
- **Hardened Docker Sandbox**:
  - No network access (`--network none`)
  - Non-root user (`--user 1000:1000`)
  - All capabilities dropped (`--cap-drop ALL`)
  - Read-only filesystem (`--read-only`)
  - Limited resources (256MB RAM, 1 CPU, 50 PIDs)
- **Zip Slip Prevention**: Validates archive entries before extraction
- **Input Validation**: Sanitizes all user inputs

---

## 📁 Project Structure

```
localCowork/
├── agent/
│   ├── cli.py                 # CLI application
│   ├── config.py              # Centralized settings
│   ├── llm/
│   │   ├── client.py          # Ollama Python library client
│   │   └── prompts.py         # LLM prompts (planner, ReAct, reflection)
│   ├── orchestrator/
│   │   ├── executor.py        # Parallel step execution
│   │   ├── planner.py         # Plan generation
│   │   ├── react_agent.py     # ReAct agentic loop
│   │   └── server.py          # FastAPI + WebSocket server
│   ├── sandbox/
│   │   └── sandbox_runner.py  # Hardened Docker execution
│   ├── security.py            # Path validation, input sanitization
│   └── tools/                 # Tool implementations
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
