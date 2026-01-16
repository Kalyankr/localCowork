from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_markdown(content: str, output: str) -> str:
    """Write content to a markdown file."""
    if not content:
        raise ValueError("Content cannot be empty")
    
    out_path = Path(output).expanduser()
    
    # Create parent directories if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_path.write_text(content)
    logger.info(f"Markdown written to {out_path}")
    return f"Markdown written to {out_path}"


def dispatch(op: str = "create", **kwargs) -> str:
    """Dispatch markdown operations."""
    if op == "create":
        content = kwargs.get("content", "")
        output = kwargs.get("output")
        if not output:
            raise ValueError("Output path is required")
        return create_markdown(content, output)
    
    raise ValueError(f"Unsupported markdown op: {op}")
