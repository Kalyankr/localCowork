from agent.llm.client import call_llm
from typing import Optional

def summarize_text(text: str) -> str:
    prompt = f"Summarize the following text concisely:\n\n{text}"
    return call_llm(prompt)

def extract_info(text: str, what: str) -> str:
    prompt = f"From the following text, extract {what}:\n\n{text}"
    return call_llm(prompt)

def transform_text(text: str, instruction: str) -> str:
    prompt = f"Transform the following text according to this instruction: {instruction}\n\n{text}"
    return call_llm(prompt)

def dispatch(op: str, **kwargs):
    # Support both 'text' and 'content' as input keys for flexibility
    text = kwargs.get("text") or kwargs.get("content") or kwargs.get("input")
    
    if not text:
        # If text is a list (e.g. from a previous step), join it
        if "files" in kwargs and isinstance(kwargs["files"], list):
            text = "\n".join(kwargs["files"])
        else:
            raise ValueError("No input text provided to text_op")

    if op == "summarize":
        return summarize_text(text)
    if op == "extract":
        return extract_info(text, kwargs.get("what", "key information"))
    if op == "transform":
        return transform_text(text, kwargs.get("instruction", "beautify"))
    
    raise ValueError(f"Unsupported text op: {op}")
