from pathlib import Path


def create_markdown(content: str, output: str):
    Path(output).expanduser().write_text(content)
    return f"Markdown written to {output}"


def dispatch(**kwargs):
    return create_markdown(kwargs["content"], kwargs["output"])
