import logging
from agent.llm.client import call_llm

logger = logging.getLogger(__name__)


def summarize_text(text: str) -> str:
    """Summarize text using the LLM."""
    if not text or not text.strip():
        return "No text to summarize."
    prompt = f"Summarize the following text concisely:\n\n{text}"
    return call_llm(prompt)


def extract_info(text: str, what: str) -> str:
    """Extract specific information from text using the LLM."""
    if not text or not text.strip():
        return "No text to extract from."
    prompt = f"From the following text, extract {what}:\n\n{text}"
    return call_llm(prompt)


def transform_text(text: str, instruction: str) -> str:
    """Transform text according to an instruction using the LLM."""
    if not text or not text.strip():
        return "No text to transform."
    prompt = f"Transform the following text according to this instruction: {instruction}\n\n{text}"
    return call_llm(prompt)


def dispatch(op: str, **kwargs) -> str:
    """Dispatch text operations."""
    # Support multiple input keys for flexibility
    text = kwargs.get("text") or kwargs.get("content") or kwargs.get("input")

    if not text:
        # If text is a list (e.g. from a previous step), join it
        files = kwargs.get("files")
        if files and isinstance(files, list):
            text = "\n".join(str(f) for f in files)

    if not text:
        raise ValueError(
            "No input text provided to text_op. Use 'text', 'content', or 'input' key."
        )

    if op == "summarize":
        return summarize_text(text)
    if op == "extract":
        return extract_info(text, kwargs.get("what", "key information"))
    if op == "transform":
        return transform_text(text, kwargs.get("instruction", "beautify"))

    raise ValueError(f"Unsupported text op: {op}")
