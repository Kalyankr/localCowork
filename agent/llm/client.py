import os


def call_llm(prompt: str) -> str:
    return """
{
  "steps": [
    {
      "id": "list_downloads",
      "description": "List files in Downloads",
      "action": "file_op",
      "args": {"op": "list", "path": "~/Downloads"},
      "depends_on": []
    }
  ]
}
"""
