from typing import Callable, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool functions that can be called by the executor."""
    
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool function."""
        logger.debug(f"Registering tool: {name}")
        self.tools[name] = func

    def get(self, name: str) -> Callable[..., Any]:
        """Get a registered tool by name."""
        if name not in self.tools:
            available = ", ".join(self.tools.keys())
            raise KeyError(f"Tool '{name}' not registered. Available: {available}")
        return self.tools[name]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools
