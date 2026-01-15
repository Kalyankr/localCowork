import json
from agent.llm.client import call_llm
from agent.orchestrator.models import Plan


PLANNER_PROMPT = """
You are a planning agent. Convert the user's request into a structured plan.

Output JSON with:
- steps: list of steps
- each step has: id, description, action, args, depends_on

Actions must be one of: ["python", "file_op"].
"""


def generate_plan(user_request: str) -> Plan:
    prompt = PLANNER_PROMPT + f"\nUser request: {user_request}"
    raw = call_llm(prompt)
    data = json.loads(raw)
    return Plan(**data)
