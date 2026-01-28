# Contributing to LocalCowork

Thank you for your interest in contributing to LocalCowork! 🎉

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (for sandboxed Python execution)
- [Ollama](https://ollama.com/) with a model installed

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Kalyankr/localCowork.git
cd localCowork

# Install dependencies with uv
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
# uv run pre-commit install

# Start Ollama (if not running)
ollama serve

# Run the agent
uv run localcowork
```

## 📝 Development Guidelines

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Hints

- Use type hints for all function parameters and return values
- Use `Optional[T]` for nullable types
- Use modern syntax: `list[str]` instead of `List[str]`

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=agent

# Run specific test file
uv run pytest tests/test_file_tools.py

# Run in verbose mode
uv run pytest -v
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Formatting, no logic change
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add image recognition tool
fix: handle timeout in LLM client
docs: update installation instructions
```

## 🏗️ Project Structure

```
localCowork/
├── agent/
│   ├── cli/           # CLI interface
│   ├── llm/           # LLM client and prompts
│   ├── orchestrator/  # ReAct agent and API server
│   ├── sandbox/       # Sandboxed code execution
│   ├── tools/         # Tool implementations
│   ├── config.py      # Configuration
│   └── security.py    # Security utilities
├── tests/             # Test files
├── main.py            # Server entry point
└── pyproject.toml     # Project configuration
```

## 🔧 Adding New Tools

1. Create a new file in `agent/tools/` (e.g., `my_tools.py`)
2. Implement your tool functions with proper type hints
3. Register tools in `agent/orchestrator/deps.py`
4. Add tests in `tests/test_my_tools.py`

Example:
```python
# agent/tools/my_tools.py
from typing import Any

def my_tool(arg1: str, arg2: int = 10) -> dict[str, Any]:
    """Description of what the tool does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Result dictionary
        
    Raises:
        MyToolError: If something goes wrong
    """
    # Implementation
    return {"result": "success"}
```

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What you expected
4. **Actual Behavior**: What actually happened
5. **Environment**: OS, Python version, Ollama model
6. **Logs**: Any relevant error messages

## 📬 Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Add/update tests
5. Run linting and tests
6. Commit with a descriptive message
7. Push to your fork
8. Open a Pull Request

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Questions?

Feel free to open an issue for questions or discussions!
