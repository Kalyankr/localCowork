"""Token counting and truncation utilities.

Provides accurate token-based context management instead of
character-based counting. Uses tiktoken for tokenization.
"""

from __future__ import annotations

import tiktoken

# Use cl100k_base as a general-purpose encoding.
# Gives reasonable estimates across models (GPT-4, Mistral, etc.)
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """Lazily initialise the shared tiktoken encoder."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text*."""
    return len(_get_encoding().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens.

    If the text fits, it is returned unchanged.
    Otherwise the token list is sliced and decoded, with an
    ellipsis appended to signal truncation.
    """
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]) + "..."
