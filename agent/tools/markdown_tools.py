from pathlib import Path


def create_markdown(content: str, output: str):
    Path(output).expanduser().write_text(content)
    return f"Markdown written to {output}"


def dispatch(op: str = "create", **kwargs):
    if op == "create":
        return create_markdown(kwargs["content"], kwargs["output"])
    raise ValueError(f"Unsupported markdown op: {op}")
