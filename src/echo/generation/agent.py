import logging
import time
from typing import Dict, Any, List, Optional

import httpx

from echo.generation.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

#Example
# SYSTEM_PROMPT = (
#     "You are a research assistant with access to tools. You MUST call tools using their "
#     "exact registered function name — never invent a tool name. Available tools will be "
#     "provided in the tools list for this request. Use them as needed to answer the user's "
#     "question accurately. You may call tools multiple times. Once you have enough "
#     "information, answer directly without calling any more tools. If you cannot find the "
#     "answer, say so honestly — do not guess."
# )

MAX_RESULT_PREVIEW = 1024


class Agent:
    def __init__(
        self,
        system_prompt: str,
        llm_api_url: str,
        tool_registry: ToolRegistry,
        token: Optional[str] = None,
        max_steps: int = 5,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        messages: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_retries = max_retries
        # The agent reaches the model only over HTTP through the LLM API
        # (`POST /chat`); it never holds an OllamaClient. A bearer `token` (the
        # caller's, in the session flow) is forwarded so the downstream /chat
        # calls authenticate as the same user.
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._http = httpx.Client(base_url=llm_api_url.rstrip("/"), timeout=timeout, headers=headers)
        # Conversation history lives on the instance. Hydrate from a persisted
        # transcript when resuming a session, otherwise start a fresh one.
        self.messages: List[Dict[str, Any]] = (
            messages if messages else [{"role": "system", "content": system_prompt}]
        )

    def _chat(self, tools: List[Dict[str, Any]] | None) -> Dict[str, Any]:
        """POST one turn to the LLM API and return the assistant message
        (content + optional tool_calls), with retries on transport errors."""
        payload = {"messages": self.messages, "tools": tools, "temperature": self.temperature}

        last_exec = None
        for attempt in range(self.max_retries):
            try:
                response = self._http.post("/chat", json=payload)
                if response.status_code != 200:
                    logger.error(f"LLM API /chat failed: {response.text}")
                response.raise_for_status()
                return response.json()["response"]

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exec = e
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait}s.")
                time.sleep(wait)

        raise RuntimeError(f"LLM API unreachable after {self.max_retries} retries") from last_exec

    def _run_step(self, tools: List[Dict[str, Any]] | None) -> Dict[str, Any]:
        """Run one LLM turn through the LLM API, append the resulting assistant
        message onto self.messages, and return it."""
        message = self._chat(tools)

        assistant_message: Dict[str, Any] = {"role": "assistant", "content": message.get("content", "")}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        self.messages.append(assistant_message)
        return assistant_message

    def send(self, query: str) -> Dict[str, Any]:
        """Run the agent loop to a final answer. One POST to the LLM API per step,
        capped at max_steps; returns the answer plus the tool calls made."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        self.messages.append({"role": "user", "content": query})
        schemas = self.tool_registry.get_schemas()
        tool_call_log: List[Dict[str, Any]] = []

        for _ in range(self.max_steps):
            message = self._run_step(schemas)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                return {"answer": message["content"], "tool_calls_made": tool_call_log}

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name")
                args = function.get("arguments", {})

                try:
                    result = str(self.tool_registry.execute(name, args))
                except Exception as e:
                    result = f"Error executing tool {name}: {e}"
                    logger.error(result)

                self.messages.append({"role": "tool", "tool_name": name, "content": result})
                tool_call_log.append({"tool": name, "args": args, "result_preview": result[:MAX_RESULT_PREVIEW]})

        logger.warning(f"Agent hit max_steps({self.max_steps}) without a final answer")
        # Final fallback turn drops tools to force a direct answer.
        message = self._run_step(None)
        return {
            "answer": message["content"],
            "tool_calls_made": tool_call_log,
            "warning": "max_steps reached",
        }

    def close(self):
        self._http.close()
