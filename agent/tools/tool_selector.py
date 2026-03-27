"""Context-aware tool suggestion engine.

Reduces prompt token usage by including only relevant tool descriptions
based on task intent classification.  Uses keyword matching (no LLM call)
for zero-latency classification.

Strategy:
 - Map tool groups to keyword patterns.
 - On each iteration, match the goal against patterns.
 - Always include tools already used in previous steps.
 - Fall back to all tools when the intent is ambiguous.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool groups — each maps a logical category to its tools + trigger patterns
# ---------------------------------------------------------------------------

_TOOL_GROUPS: dict[str, dict[str, object]] = {
    "file_ops": {
        "tools": ["read_file", "write_file", "edit_file"],
        "patterns": [
            r"\bfiles?\b",
            r"\bread\b",
            r"\bwrite\b",
            r"\bedit\b",
            r"\bcreate\b.*\b(?:file|script|config|code)\b",
            r"\bmodify\b",
            r"\bupdate\b.*\b(?:file|config|code)\b",
            r"\bshow\s+me\b",
            r"\bdisplay\b.*\bcontent",
            r"\bconfig\b",
            r"\bsave\b",
            r"\bopen\b",
            r"\.\w{1,5}\b",  # file extensions like .py, .json, .yaml
        ],
    },
    "shell": {
        "tools": ["shell"],
        "patterns": [
            r"\brun\b",
            r"\bexecute\b",
            r"\binstall\b",
            r"\blist\b",
            r"\bfind\b",
            r"\bmove\b",
            r"\bcopy\b",
            r"\bdelete\b",
            r"\bremove\b",
            r"\brename\b",
            r"\bdirector(?:y|ies)\b",
            r"\bfolders?\b",
            r"\bgit\b",
            r"\bnpm\b",
            r"\bpip\b",
            r"\bdocker\b",
            r"\bmake\b",
            r"\bbuild\b",
            r"\bcompile\b",
            r"\btests?\b",
            r"\bgrep\b",
            r"\bls\b",
            r"\bmkdir\b",
            r"\bcurl\b",
            r"\bprocess\b",
            r"\bkill\b",
            r"\bservice\b",
            r"\bdisk\b",
            r"\bsystem\b",
            r"\bcommand\b",
            r"\bterminal\b",
        ],
    },
    "code": {
        "tools": ["python"],
        "patterns": [
            r"\bpython\b",
            r"\bscript\b",
            r"\banalyze?\b",
            r"\bdata\b",
            r"\bcalculat\w*\b",
            r"\bcomput\w*\b",
            r"\bparse\b",
            r"\btransform\b",
            r"\bcsv\b",
            r"\bpandas\b",
            r"\bchart\b",
            r"\bplot\b",
            r"\bgraph\b",
            r"\bstatistic\w*\b",
            r"\bsummariz\w*\b.*\b(?:data|csv|file)\b",
            r"\bcount\b.*\b(?:lines?|rows?|words?)\b",
            r"\bconvert\b",
        ],
    },
    "web": {
        "tools": ["web_search", "fetch_webpage"],
        "patterns": [
            r"\bsearch\b(?!.*\bfile)",  # "search" but not "search file"
            r"\bweb\b",
            r"\binternet\b",
            r"\bonline\b",
            r"\burl\b",
            r"\bhttps?://",
            r"\bwebsite\b",
            r"\bbrowse\b",
            r"\bfetch\b.*\bpage\b",
            r"\bdownload\b.*\bfrom\b",
            r"\bdocumentation\b",
            r"\btutorial\b",
            r"\bhow\s+to\b",
            r"\bwhat\s+is\b",
            r"\blook\s*up\b",
            r"\bfind\s+out\b",
            r"\bgoogle\b",
            r"\blatest\b",
        ],
    },
    "memory": {
        "tools": ["memory_store", "memory_recall"],
        "patterns": [
            r"\bremember\b",
            r"\brecall\b",
            r"\bmemor(?:y|ies|ize)\b",
            r"\bforget\b",
            r"\bpreference\b",
            r"\byou\s+know\b",
            r"\bdo\s+you\s+remember\b",
            r"\bwhat\s+do\s+you\s+know\b",
            r"\bkeep\s+in\s+mind\b",
            r"\bstore\b.*\bfact\b",
        ],
    },
}

# Compiled patterns (built once at import time)
_COMPILED_GROUPS: dict[str, tuple[list[str], list[re.Pattern[str]]]] = {}

for _name, _info in _TOOL_GROUPS.items():
    _COMPILED_GROUPS[_name] = (
        list(_info["tools"]),  # type: ignore[arg-type]
        [re.compile(p, re.IGNORECASE) for p in _info["patterns"]],  # type: ignore[union-attr]
    )

# Core tools always included (shell is almost universally useful)
_ALWAYS_INCLUDE: frozenset[str] = frozenset({"shell"})


def suggest_tools(
    goal: str,
    available_tools: Sequence[str],
    *,
    used_tools: Sequence[str] = (),
) -> list[str]:
    """Return the subset of available tools relevant to *goal*.

    Args:
        goal: The user's task description.
        available_tools: Names of all registered tools.
        used_tools: Tools already used in earlier steps (always retained).

    Returns:
        Sorted list of tool names to include in the prompt.
    """
    all_tools = set(available_tools)

    if not goal.strip():
        return sorted(all_tools)

    matched: set[str] = set(_ALWAYS_INCLUDE)
    matched_groups: set[str] = set()

    # Match goal against compiled patterns
    for group_name, (tools, patterns) in _COMPILED_GROUPS.items():
        for pat in patterns:
            if pat.search(goal):
                matched.update(tools)
                matched_groups.add(group_name)
                break  # one match per group is enough

    # Always keep tools that were already used in prior steps
    matched.update(used_tools)

    # Ambiguous / complex goal → return all tools
    # (matched 0 groups → no clear intent, 4+ → covers almost everything)
    if len(matched_groups) == 0 or len(matched_groups) >= 4:
        logger.debug(
            "tool_selector_all",
            reason="ambiguous" if not matched_groups else "complex",
            matched_groups=len(matched_groups),
        )
        return sorted(all_tools)

    # Intersect with available tools
    result = sorted(matched & all_tools)

    # If filtering left fewer than 2 tools, include all (safety net)
    if len(result) < 2:
        return sorted(all_tools)

    logger.debug(
        "tool_selector_filtered",
        goal=goal[:80],
        groups=sorted(matched_groups),
        tools=result,
        saved=len(all_tools) - len(result),
    )
    return result
