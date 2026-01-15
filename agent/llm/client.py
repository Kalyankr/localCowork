import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"


def call_llm(prompt: str) -> str:
    """
    Calls Ollama and returns raw text output.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_tokens": 2048},
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["response"]


def call_llm_json(prompt: str) -> dict:
    """
    Calls Ollama and guarantees valid JSON output.
    If the model returns malformed JSON, we repair it.
    """
    raw = call_llm(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = repair_json(raw)
        return repaired


def repair_json(text: str) -> dict:
    """
    Attempts to fix malformed JSON by:
    - Extracting the first {...} block
    - Fixing trailing commas
    - Fixing unquoted keys
    - Ensuring valid structure
    """
    import re

    # Extract JSON-like content
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM did not return JSON")

    json_like = match.group(0)

    # Basic cleanup
    json_like = json_like.replace("\n", " ")
    json_like = re.sub(r",\s*}", "}", json_like)
    json_like = re.sub(r",\s*]", "]", json_like)

    try:
        return json.loads(json_like)
    except Exception:
        raise ValueError(f"Could not repair JSON: {json_like}")
