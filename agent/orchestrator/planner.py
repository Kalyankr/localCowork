from agent.llm.client import call_llm_json
from agent.llm.prompts import PLANNER_PROMPT
from agent.orchestrator.models import Plan


def generate_plan(user_request: str) -> Plan:
    prompt = PLANNER_PROMPT + user_request
    data = call_llm_json(prompt)
    return Plan(**data)
