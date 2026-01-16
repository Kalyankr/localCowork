# LocalCowork - Technical Deep Dive

This document provides a comprehensive explanation of the LocalCowork architecture, inspired by [Claude Cowork](https://support.claude.com/en/articles/13345190-getting-started-with-cowork).

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Execution Flow](#execution-flow)
5. [Tool System](#tool-system)
6. [Sandbox Security](#sandbox-security)
7. [LLM Integration](#llm-integration)
8. [CLI Interface](#cli-interface)
9. [API Server](#api-server)

---

## Overview

LocalCowork is a **CLI-based agentic AI assistant** that transforms natural language requests into executable multi-step plans. Unlike traditional chatbots that respond to one prompt at a time, LocalCowork:

1. **Analyzes** your request
2. **Plans** a sequence of steps (as a DAG)
3. **Executes** steps with dependency resolution
4. **Reports** results with a friendly summary

### Key Differentiators from Claude Cowork

| Feature | Claude Cowork | LocalCowork |
|---------|---------------|-------------|
| LLM | Claude API (cloud) | Ollama (local) |
| Interface | Desktop GUI | CLI |
| Sandbox | VM-based | Docker container |
| Cost | Subscription | Free (local compute) |
| Privacy | Cloud processing | 100% local |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
│            "Organize my downloads by file type"                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                             │
│  • Parse arguments (--yes, --dry-run, --verbose)                │
│  • Display plan with confirmation                                │
│  • Show live progress table                                      │
│  • Render final results                                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLANNER (planner.py)                        │
│  • Sends request + tool schema to LLM                           │
│  • Receives JSON plan with steps                                 │
│  • Validates and parses into Plan model                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTOR (executor.py)                       │
│  • Builds dependency graph                                       │
│  • Runs steps in waves (parallel when possible)                 │
│  • Resolves variable interpolation                               │
│  • Stores outputs in context for dependent steps                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   TOOL REGISTRY  │ │     SANDBOX      │ │   LLM CLIENT     │
│                  │ │                  │ │                  │
│  file_op         │ │  Docker-based    │ │  Ollama API      │
│  markdown_op     │ │  Python runner   │ │  JSON repair     │
│  pdf_op          │ │  Network: none   │ │  Error handling  │
│  data_op         │ │  Isolated fs     │ │                  │
│  text_op         │ │                  │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## Core Components

### 1. Models (`agent/orchestrator/models.py`)

Pydantic models define the data structures:

```python
class Step:
    id: str              # Unique identifier (e.g., "list_all")
    description: str     # Human-readable description
    action: str          # Tool to use ("file_op", "python", etc.)
    args: Dict[str, Any] # Arguments for the tool
    depends_on: List[str] # Step IDs that must complete first

class Plan:
    steps: List[Step]    # Ordered list of steps

class StepResult:
    step_id: str
    status: str          # "success", "error", "skipped"
    output: Any          # Result data
    error: str           # Error message if failed
```

### 2. Planner (`agent/orchestrator/planner.py`)

The planner converts natural language to a structured plan:

```python
def generate_plan(user_request: str) -> Plan:
    prompt = PLANNER_PROMPT + user_request
    data = call_llm_json(prompt)  # LLM returns JSON
    return Plan(**data)
```

The `PLANNER_PROMPT` contains:
- Tool schema (available operations)
- Examples of good plans
- Rules for JSON formatting
- Dependency guidelines

### 3. Executor (`agent/orchestrator/executor.py`)

The executor runs the plan with parallel execution:

```python
class Executor:
    async def run(self) -> dict:
        # Wave-based execution
        while not all_complete:
            ready_steps = get_steps_with_deps_met()
            
            if parallel and len(ready_steps) > 1:
                # Run independent steps concurrently
                await asyncio.gather(*[run_step(s) for s in ready_steps])
            else:
                # Sequential execution
                for step in ready_steps:
                    await run_step(step)
```

Key features:
- **Dependency resolution**: Steps wait for their dependencies
- **Context propagation**: Output from step A is available to step B
- **Variable interpolation**: `"path": "list_all"` resolves to actual file list
- **Progress callbacks**: CLI receives real-time status updates

### 4. Tool Registry (`agent/orchestrator/tool_registry.py`)

A simple registry pattern for tool dispatch:

```python
class ToolRegistry:
    def register(self, name: str, func: Callable)
    def get(self, name: str) -> Callable
    def list_tools(self) -> List[str]
    def has(self, name: str) -> bool
```

---

## Execution Flow

### Example: "Organize downloads into Images and PDFs"

```
User Request
     │
     ▼
┌─────────────────────────────────────┐
│ LLM generates plan:                 │
│                                     │
│ Step 1: list_all (file_op:list)    │
│    └─► Step 2: categorize (python) │
│           ├─► Step 3: move_imgs    │
│           └─► Step 4: move_pdfs    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Executor runs:                      │
│                                     │
│ Wave 1: [list_all]                 │
│    Output: [{path, name, ...}, ...]│
│                                     │
│ Wave 2: [categorize]               │
│    Code: imgs = [f for f in ...]   │
│    Output: {imgs: [...], pdfs: [...]}│
│                                     │
│ Wave 3: [move_imgs, move_pdfs]     │  ← Parallel!
│    Both run simultaneously          │
└─────────────────────────────────────┘
     │
     ▼
Results + Summary
```

### Variable Resolution

The executor supports several interpolation patterns:

```python
# Direct reference
"path": "list_all"  →  [{...}, {...}]

# Dictionary access
"src": "categorize['imgs']"  →  [img files]

# Path concatenation
"dest": "base_dir/Images"  →  "/home/user/Downloads/Images"
```

---

## Tool System

### Available Tools

| Tool | Operations | Description |
|------|------------|-------------|
| `file_op` | list, move, mkdir, read, write, rename | File system operations |
| `markdown_op` | create | Generate markdown files |
| `pdf_op` | extract | Extract PDF metadata |
| `data_op` | csv_to_excel | Data format conversion |
| `text_op` | summarize, extract, transform | LLM-powered text processing |
| `python` | (code execution) | Run Python in sandbox |

### Adding a New Tool

1. Create `agent/tools/my_tool.py`:
```python
def my_operation(arg1: str, arg2: int) -> str:
    # Implementation
    return result

def dispatch(op: str, **kwargs):
    if op == "my_op":
        return my_operation(kwargs["arg1"], kwargs["arg2"])
    raise ValueError(f"Unknown op: {op}")
```

2. Register in `agent/tools/__init__.py`:
```python
def create_default_registry():
    registry.register("my_op", my_tool.dispatch)
```

3. Add to `PLANNER_PROMPT` in `agent/llm/prompts.py`:
```python
- "my_op"
    - args: {"op": "my_op", "arg1": str, "arg2": int}
```

---

## Sandbox Security

### Docker Isolation

Python code from plans runs in a secure Docker container:

```python
cmd = [
    "docker", "run", "--rm",
    "--network", "none",      # No internet access
    "-v", f"{tmpdir}:/app",   # Only mount temp directory
    "python:3.12-slim",
    "python", "script.py"
]
```

Security measures:
- **No network**: `--network none` prevents data exfiltration
- **Isolated filesystem**: Only temp directory is mounted
- **Timeout**: Execution limited to 30 seconds
- **Read-only context**: Variables injected, not actual file access

### Context Injection

The sandbox receives variables from previous steps:

```python
# Injected at top of script
list_all = [{"path": "/home/user/file.txt", ...}]
categorize = None

# User's code
imgs = [f for f in list_all if f['name'].endswith('.jpg')]

# Capture results (appended automatically)
print(f'__RESULT__:' + json.dumps(categorize))
print(f'__TRACE_VARS__:' + json.dumps(locals()))
```

---

## LLM Integration

### Ollama Configuration

```python
# Environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
```

### JSON Repair

Local LLMs sometimes produce malformed JSON. The `repair_json()` function handles:
- Trailing commas
- Unquoted values
- Literal newlines in strings
- Missing brackets

```python
def call_llm_json(prompt: str) -> dict:
    raw = call_llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return repair_json(raw)  # Attempt to fix
```

---

## CLI Interface

### Commands

```bash
# Run a task
localcowork run "organize my downloads" [options]

# Start API server
localcowork serve --host 127.0.0.1 --port 8000
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--yes` | `-y` | Skip confirmation |
| `--dry-run` | `-n` | Show plan only |
| `--no-parallel` | `-s` | Sequential execution |
| `--verbose` | `-v` | Debug logging |

### Progress Display

```
 Step                 Status       Description
 list_all             ✓ done       file_op
 categorize           ▶ running    Filter files into categories
 move_imgs            ⏳ pending   file_op
 move_pdfs            ⏳ pending   file_op
```

---

## API Server

FastAPI server for programmatic access:

```python
# POST /tasks
{
    "request": "organize my downloads"
}

# Response
{
    "task_id": "uuid",
    "plan": {...},
    "results": {...}
}

# GET /health
{"status": "ok"}
```

### Query Parameters

- `parallel=true|false` - Enable/disable parallel execution

---

## Directory Structure

```
localCowork/
├── main.py                 # Entry point for uvicorn
├── pyproject.toml          # Project dependencies
├── agent/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI application
│   ├── llm/
│   │   ├── client.py       # Ollama API client
│   │   └── prompts.py      # System prompts for planner
│   ├── orchestrator/
│   │   ├── executor.py     # Plan execution engine
│   │   ├── models.py       # Pydantic data models
│   │   ├── planner.py      # LLM-based plan generation
│   │   ├── server.py       # FastAPI server
│   │   └── tool_registry.py # Tool registration
│   ├── sandbox/
│   │   └── sandbox_runner.py # Docker-based code execution
│   └── tools/
│       ├── file_tools.py   # File operations
│       ├── markdown_tools.py
│       ├── pdf_tools.py
│       ├── data_tools.py
│       └── text_tools.py
└── docs/
    ├── progress.md         # Enhancement tracking
    └── project_explained.md # This file
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework |
| `rich` | Terminal UI (tables, progress, panels) |
| `pydantic` | Data validation |
| `requests` | HTTP client for Ollama |
| `fastapi` | API server |
| `uvicorn` | ASGI server |
| `pandas` | Data operations |
| `pypdf` | PDF processing |

---

## Running Locally

### Prerequisites

1. **Python 3.12+**
2. **Docker** (for sandbox)
3. **Ollama** with a model installed

### Setup

```bash
# Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# Clone and install
git clone https://github.com/yourusername/localCowork.git
cd localCowork
uv pip install -e .

# Run
localcowork run "list files in my downloads"
```

---

## Design Principles

1. **Privacy First**: All processing happens locally
2. **Transparency**: Users see the plan before execution
3. **Safety**: Sandboxed execution for untrusted code
4. **Extensibility**: Easy to add new tools
5. **Reliability**: Error handling and JSON repair for local LLMs
