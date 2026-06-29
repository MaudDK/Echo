import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, schema: dict, handler: Callable[..., str]) -> None:
        if name in self._tools:
            logger.warning(f"Tool '{name}' is already registered — overwriting")
        self._tools[name] = {"schema": schema, "handler": handler}
        logger.info(f"Registered tool: '{name}'")

    def get_schemas(self) -> List[dict]:
        return [entry["schema"] for entry in self._tools.values()]

    def execute(self, name: str, args: dict) -> str:
        entry = self._tools.get(name)
        if entry is None:
            logger.warning(f"LLM attempted to call unknown tool: '{name}'")
            return f"Error: unknown tool '{name}'. Available tools: {list(self._tools.keys())}"

        try:
            return entry["handler"](**args)
        except Exception as e:
            logger.exception(f"Tool '{name}' execution failed with args {args}")
            return f"Error executing '{name}': {e}"

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())