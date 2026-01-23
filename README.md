<h1 align="center">
  <br>
  🤖 LocalCowork
  <br>
</h1>

<h4 align="center">A privacy-first AI assistant that runs entirely on your machine.</h4>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/LLM-Ollama-orange.svg" alt="Ollama">
</p>

---

**LocalCowork** transforms natural language requests into executable multi-step plans. Inspired by [Claude Cowork](https://support.claude.com/en/articles/13345190-getting-started-with-cowork), it brings agentic AI capabilities to your terminal—100% locally, 100% private.

```bash
$ localcowork run "Organize my downloads by file type"

📋 Plan
┌────────────────────────────────────────────────────┐
│  1. list_all → List files in ~/Downloads          │
│  2. categorize → Group by extension               │
│  3. move_imgs → Move images to Images/            │
│  4. move_pdfs → Move PDFs to Documents/           │
└────────────────────────────────────────────────────┘

Execute this plan? [Y/n]: y

 Step              Status       Description
 list_all          ✓ done       file_op
 categorize        ✓ done       Filter files
 move_imgs         ✓ done       file_op
 move_pdfs         ✓ done       file_op

Completed: 4 succeeded, 0 failed
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🔒 **Privacy First** | All processing happens locally. Your files never leave your machine. |
| ⚡ **Parallel Execution** | Independent steps run concurrently for faster results. |
| 🔍 **Transparent Plans** | See exactly what will happen before execution. |
| 🐳 **Sandboxed Code** | Python code runs in isolated Docker containers. |
| 📊 **Live Progress** | Real-time status updates for each step. |
| 🛠️ **Extensible Tools** | File ops, PDF extraction, data conversion, and more. |

---

## Installation

### Prerequisites

- **Python 3.12+**
- **Docker** (for sandboxed Python execution)
- **Ollama** with a model installed

### Quick Start

```bash
# 1. Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# 2. Clone the repository
git clone https://github.com/yourusername/localCowork.git
cd localCowork

# 3. Install with uv (recommended) or pip
uv pip install -e .
# or: pip install -e .

# 4. Run your first task
localcowork run "List all PDF files in my downloads"
```

---

## Usage

### Basic Commands

```bash
# Run a task with confirmation prompt
localcowork run "organize my downloads folder"

# Skip confirmation (for automation)
localcowork run "move all images to Pictures" --yes

# Preview plan without executing
localcowork run "delete temporary files" --dry-run

# Run sequentially (disable parallel execution)
localcowork run "process files" --no-parallel

# Enable debug logging
localcowork run "analyze documents" --verbose
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--yes` | `-y` | Skip confirmation prompt |
| `--dry-run` | `-n` | Show plan without executing |
| `--no-parallel` | `-s` | Run steps sequentially |
| `--verbose` | `-v` | Enable debug output |

### API Server

```bash
# Start the REST API server
localcowork serve --host 127.0.0.1 --port 8000
```

```bash
# Make a request
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"request": "list files in downloads"}'
```

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Your Request   │────▶│   LLM Planner    │────▶│    Executor      │
│                  │     │   (Ollama)       │     │   (Parallel)     │
│ "organize files" │     │   JSON Plan      │     │   Run Steps      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                              ┌─────────────────────────────┤
                              ▼                             ▼
                    ┌──────────────────┐         ┌──────────────────┐
                    │   Tool Registry  │         │     Sandbox      │
                    │   file_op, etc   │         │   Docker Python  │
                    └──────────────────┘         └──────────────────┘
```

1. **Plan Generation**: Your request is sent to a local LLM (via Ollama) which generates a structured plan with dependencies.

2. **Dependency Resolution**: The executor builds a DAG and runs steps in waves—parallel when dependencies allow.

3. **Tool Execution**: Built-in tools handle file operations, PDF processing, data conversion, and more.

4. **Sandboxed Code**: Python code runs in isolated Docker containers with no network access.

5. **Results**: A friendly summary shows what was accomplished.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `mistral` | Model to use for planning |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |

```bash
# Example: Use a different model
export OLLAMA_MODEL=llama3
localcowork run "summarize my notes"
```

---

## Available Tools

| Tool | Operations | Example |
|------|------------|---------|
| `file_op` | list, move, mkdir, read, write | File management |
| `pdf_op` | extract metadata | PDF analysis |
| `data_op` | csv_to_excel | Data conversion |
| `markdown_op` | create | Document generation |
| `text_op` | summarize, extract, transform | LLM-powered text processing |
| `python` | Custom code execution | Complex logic (sandboxed) |

---

## Project Structure

```
localCowork/
├── agent/
│   ├── cli.py              # CLI application
│   ├── llm/                # LLM client & prompts
│   ├── orchestrator/       # Planner, executor, models
│   ├── sandbox/            # Docker-based code runner
│   └── tools/              # Built-in tool implementations
├── docs/
│   ├── progress.md         # Enhancement roadmap
│   └── project_explained.md # Technical deep-dive
├── main.py                 # API server entry point
├── pyproject.toml          # Dependencies
└── README.md
```

---

## Contributing

Contributions are welcome! Please read the [project documentation](docs/project_explained.md) to understand the architecture.

```bash
# Run tests
pytest

# Check code style
ruff check .

# Format code
ruff format .
```

---

<p align="center">
  Made for local-first AI
</p>
