from agent.llm.client import call_llm_json, call_llm
from agent.llm.prompts import PLANNER_PROMPT, SUMMARIZER_PROMPT
from agent.orchestrator.models import Plan


def generate_plan(user_request: str) -> Plan:
    prompt = PLANNER_PROMPT + user_request
    data = call_llm_json(prompt)
    return Plan(**data)


def summarize_results(user_request: str, results: dict) -> str:
    res_str = ""
    for step_id, res in results.items():
        # Clean up output for the summarizer (remove massive trace data)
        clean_output = str(res.output)
        if "__TRACE_VARS__" in clean_output:
            clean_output = clean_output.split("__TRACE_VARS__")[0].strip()
        
        status_line = f"Step {step_id}: {res.status}"
        if clean_output and clean_output != "None":
            status_line += f" (Output: {clean_output[:200]})"
        if res.error:
            status_line += f" (Error: {res.error})"
        
        res_str += status_line + "\n"
    
    prompt = SUMMARIZER_PROMPT.format(request=user_request, results=res_str)
    return call_llm(prompt)
