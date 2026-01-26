"""LLM client using official Ollama Python library.

This module provides a clean interface to Ollama using the official library
instead of raw HTTP requests. Benefits:
- Cleaner API with typed responses
- Built-in connection management
- Async support ready
- Better error handling
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any

import ollama
from ollama import ResponseError, RequestError

from agent.config import settings

logger = logging.getLogger(__name__)

# Configuration from centralized settings
MODEL = settings.ollama_model
TIMEOUT = settings.ollama_timeout
MAX_JSON_RETRIES = settings.max_json_retries


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


# Create a client instance (reusable, connection pooled)
_client: Optional[ollama.Client] = None


def _get_client() -> ollama.Client:
    """Get or create the Ollama client singleton."""
    global _client
    if _client is None:
        # Extract host from the old URL format if needed
        host = settings.ollama_url.replace("/api/generate", "").replace("/api/chat", "")
        if host.endswith("/"):
            host = host[:-1]
        _client = ollama.Client(host=host, timeout=TIMEOUT)
    return _client


def call_llm(prompt: str, force_json: bool = False) -> str:
    """
    Calls Ollama and returns raw text output.
    
    Args:
        prompt: The prompt to send to the model
        force_json: If True, use Ollama's JSON mode to force valid JSON output
        
    Returns:
        The model's response text
        
    Raises:
        LLMError: If the LLM request fails.
    """
    try:
        client = _get_client()
        logger.debug(f"Calling LLM with model={MODEL}, force_json={force_json}")
        
        kwargs = {
            "model": MODEL,
            "prompt": prompt,
            "options": {"num_predict": settings.max_tokens},
        }
        
        # Use Ollama's native JSON mode if requested
        if force_json:
            kwargs["format"] = "json"
        
        response = client.generate(**kwargs)
        
        return response.response
        
    except RequestError as e:
        raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
    except ResponseError as e:
        raise LLMError(f"Ollama error: {e}")
    except Exception as e:
        raise LLMError(f"LLM request failed: {e}")


def call_llm_chat(messages: List[Dict[str, str]]) -> str:
    """
    Calls Ollama with chat messages format.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        The assistant's response content
        
    Raises:
        LLMError: If the LLM request fails.
    """
    try:
        client = _get_client()
        logger.debug(f"Calling LLM chat with model={MODEL}, {len(messages)} messages")
        
        response = client.chat(
            model=MODEL,
            messages=messages,
            options={"num_predict": settings.max_tokens},
        )
        
        return response.message.content
        
    except RequestError as e:
        raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
    except ResponseError as e:
        raise LLMError(f"Ollama error: {e}")
    except Exception as e:
        raise LLMError(f"LLM chat request failed: {e}")


def call_llm_json(prompt: str, retry_prompt: str = None) -> dict:
    """
    Calls Ollama and guarantees valid JSON output.
    Uses Ollama's native JSON mode for reliable structured output.
    Retries with repair logic if JSON parsing still fails.
    
    Args:
        prompt: The initial prompt to send
        retry_prompt: Optional simpler prompt to use on retry
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        LLMError: If JSON parsing fails after all retries
    """
    last_error = None
    
    for attempt in range(MAX_JSON_RETRIES + 1):
        try:
            current_prompt = prompt
            if attempt > 0:
                logger.info(f"JSON retry attempt {attempt + 1}/{MAX_JSON_RETRIES + 1}")
                current_prompt = prompt + "\n\nREMINDER: Output ONLY valid JSON. No markdown, no code blocks, no explanation. Start with { and end with }."
            
            # Use Ollama's native JSON mode for reliable output
            raw = call_llm(current_prompt, force_json=True)
            
            # Try direct parse first
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try repair as fallback
                return repair_json(raw)
                
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_JSON_RETRIES:
                continue
            else:
                raise LLMError(
                    f"Failed to get valid JSON after {MAX_JSON_RETRIES + 1} attempts. "
                    "The AI model may be having trouble understanding the request. "
                    "Please try rephrasing."
                )


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
            json_like = text[start_idx:] + "}"
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
            if not (val.startswith('"') or val.startswith("'") or 
                    val.startswith("[") or val.startswith("{") or 
                    val.isdigit() or val in ["true", "false", "null"]):
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


def list_models() -> List[str]:
    """List available models from Ollama.
    
    Returns:
        List of model names
    """
    try:
        client = _get_client()
        response = client.list()
        return [model.model for model in response.models]
    except Exception as e:
        logger.warning(f"Failed to list models: {e}")
        return []


def check_model_exists(model_name: str = None) -> bool:
    """Check if a model exists in Ollama.
    
    Args:
        model_name: Model to check, defaults to configured model
        
    Returns:
        True if model exists
    """
    model = model_name or MODEL
    try:
        models = list_models()
        # Check for exact match or match without tag
        return any(
            m == model or m.split(":")[0] == model.split(":")[0]
            for m in models
        )
    except Exception:
        return False
