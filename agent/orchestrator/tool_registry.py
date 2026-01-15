from typing import Callable, Dict, Any


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]):
        self.tools[name] = func

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self.tools[name]
