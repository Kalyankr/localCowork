import json
import requests
import re

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
    """
    raw = call_llm(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return repair_json(raw)


def repair_json(text: str) -> dict:
    """
    Attempts to fix common LLM JSON errors:
    - Literal newlines within string values
    - Python-style string concatenation
    - Improperly quoted values
    - Missing/extra braces/brackets
    """
    # 1. Extract the cleanest JSON-like block
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found")
    
    # Use a simple stack-based approach to find the end of the object
    stack = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        json_like = text[start_idx:]
    else:
        json_like = text[start_idx:end_idx]

    # 2. Fix literal newlines inside string values
    # We look for content between double quotes and replace real newlines with \n
    new_json = ""
    in_string = False
    escape = False
    for char in json_like:
        if char == '"' and not escape:
            in_string = not in_string
            new_json += char
        elif char == '\\' and in_string and not escape:
            escape = True
            new_json += char
        elif char == '\n' and in_string:
            new_json += "\\n"
        else:
            new_json += char
            escape = False
    json_like = new_json

    # 3. Fix common syntax errors
    json_like = re.sub(r",\s*}", "}", json_like)
    json_like = re.sub(r",\s*]", "]", json_like)

    try:
        return json.loads(json_like)
    except Exception:
        def quote_val(m):
            val = m.group(2).strip()
            if not (val.startswith('"') or val.startswith("'") or val.startswith("[") or val.startswith("{") or val.isdigit() or val in ["true", "false", "null"]):
                return f'"{m.group(1)}": "{val}"'
            return m.group(0)

        fixed = re.sub(r'"(\w+)":\s*([^,\}\n]+)', quote_val, json_like)
        try:
            return json.loads(fixed)
        except:
            raise ValueError(f"Could not repair JSON: {json_like}")
