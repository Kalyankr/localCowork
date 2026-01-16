import json
import os
import logging
import requests
import re
from requests.exceptions import ConnectionError, Timeout, RequestException

logger = logging.getLogger(__name__)

# Configuration via environment variables with sensible defaults
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Retry settings for JSON generation
MAX_JSON_RETRIES = 2


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


def call_llm(prompt: str) -> str:
    """
    Calls Ollama and returns raw text output.
    
    Raises:
        LLMError: If the LLM request fails.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 2048},
    }

    try:
        logger.debug(f"Calling LLM with model={MODEL}")
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data["response"]
    except ConnectionError:
        raise LLMError(f"Cannot connect to Ollama at {OLLAMA_URL}. Is it running?")
    except Timeout:
        raise LLMError(f"LLM request timed out after {TIMEOUT}s")
    except RequestException as e:
        raise LLMError(f"LLM request failed: {e}")


def call_llm_json(prompt: str, retry_prompt: str = None) -> dict:
    """
    Calls Ollama and guarantees valid JSON output.
    Retries with a simpler prompt if JSON parsing fails.
    
    Args:
        prompt: The initial prompt to send
        retry_prompt: Optional simpler prompt to use on retry
    """
    last_error = None
    
    for attempt in range(MAX_JSON_RETRIES + 1):
        try:
            # On retry, add a hint about the previous failure
            current_prompt = prompt
            if attempt > 0:
                logger.info(f"JSON retry attempt {attempt + 1}/{MAX_JSON_RETRIES + 1}")
                # Add explicit JSON reminder
                current_prompt = prompt + "\n\nREMINDER: Output ONLY valid JSON. No markdown, no code blocks, no explanation. Start with { and end with }."
            
            raw = call_llm(current_prompt)
            
            # Try direct parse first
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try repair
                return repair_json(raw)
                
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_JSON_RETRIES:
                continue
            else:
                raise LLMError(f"Failed to get valid JSON after {MAX_JSON_RETRIES + 1} attempts. The AI model may be having trouble understanding the request. Please try rephrasing.")


def repair_json(text: str) -> dict:
    """
    Attempts to fix common LLM JSON errors:
    - Literal newlines within string values
    - Python-style string concatenation
    - Improperly quoted values
    - Missing/extra braces/brackets
    - Markdown code blocks
    """
    original_text = text
    
    # 0. Remove markdown code blocks if present
    if "```json" in text:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    elif "```" in text:
        text = re.sub(r"```\w*\s*", "", text)
        text = re.sub(r"```\s*", "", text)
    
    # 1. Extract the cleanest JSON-like block
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in response")
    
    # Use a simple stack-based approach to find the end of the object
    stack = 0
    end_idx = -1
    in_string = False
    escape = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == '{':
                stack += 1
            elif char == '}':
                stack -= 1
                if stack == 0:
                    end_idx = i + 1
                    break
        
        escape = False
    
    if end_idx == -1:
        # Try to find a closing brace anyway
        last_brace = text.rfind('}')
        if last_brace > start_idx:
            end_idx = last_brace + 1
        else:
            json_like = text[start_idx:] + "}"  # Add missing closing brace
            end_idx = len(json_like) + start_idx
    
    json_like = text[start_idx:end_idx] if end_idx != -1 else text[start_idx:]

    # 2. Fix literal newlines inside string values
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
        elif char == '\t' and in_string:
            new_json += "\\t"
        else:
            new_json += char
            escape = False
    json_like = new_json

    # 3. Fix common syntax errors
    json_like = re.sub(r",\s*}", "}", json_like)  # Trailing comma before }
    json_like = re.sub(r",\s*]", "]", json_like)  # Trailing comma before ]
    json_like = re.sub(r"'\s*:", '":', json_like)  # Single quotes for keys
    json_like = re.sub(r":\s*'([^']*)'", r': "\1"', json_like)  # Single quotes for values

    try:
        return json.loads(json_like)
    except json.JSONDecodeError:
        pass
    
    # 4. Try to fix unquoted values
    try:
        def quote_val(m):
            val = m.group(2).strip()
            if not (val.startswith('"') or val.startswith("'") or val.startswith("[") or val.startswith("{") or val.isdigit() or val in ["true", "false", "null"]):
                return f'"{m.group(1)}": "{val}"'
            return m.group(0)

        fixed = re.sub(r'"(\w+)":\s*([^,\}\]\n]+)', quote_val, json_like)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # 5. Last resort - try to extract just the steps array
    try:
        steps_match = re.search(r'"steps"\s*:\s*\[(.*?)\]', json_like, re.DOTALL)
        if steps_match:
            return {"steps": json.loads(f"[{steps_match.group(1)}]")}
    except:
        pass
    
    raise ValueError(f"Could not parse response as JSON")
