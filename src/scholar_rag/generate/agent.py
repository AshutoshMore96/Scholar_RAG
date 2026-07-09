"""
Optional ReAct agent for multi-hop literature questions.

The agent can:
  - search_papers(query)   → retrieve passages from Qdrant
  - expand_citation(pid)   → fetch full citation context for a paper id
  - fetch_abstract(pid)    → return cached abstract

This enables multi-hop reasoning: "what papers does paper X cite that are
also relevant to topic Y?"
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import httpx
from loguru import logger


_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

REACT_SYSTEM = """\
You are a research assistant that uses tools to answer multi-hop literature questions.

Available tools:
- search_papers(query: str)       → returns up to 5 relevant passages
- expand_citation(paper_id: str)  → returns citation context for a paper
- fetch_abstract(paper_id: str)   → returns the abstract of a paper

Use this format EXACTLY:
Thought: <your reasoning>
Action: <tool_name>(<arguments>)
Observation: <result will be filled in>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information.
Final Answer: <your cited literature review answer>
"""


class ReActLiteratureAgent:
    def __init__(
        self,
        search_fn: Callable[[str], list[dict]],
        expand_fn: Callable[[str], str],
        abstract_fn: Callable[[str], str],
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        max_steps: int = 6,
    ) -> None:
        self.search = search_fn
        self.expand = expand_fn
        self.abstract = abstract_fn
        self.model = model
        self.url = ollama_url.rstrip("/")
        self.max_steps = max_steps
        self._client = httpx.Client(timeout=120.0)

    def run(self, question: str) -> str:
        trajectory = [REACT_SYSTEM, f"\nQuestion: {question}\n"]
        for step in range(self.max_steps):
            prompt = "\n".join(trajectory)
            response = self._llm(prompt)
            trajectory.append(response)

            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()

            action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", response)
            if not action_match:
                break

            tool_name = action_match.group(1)
            raw_arg = action_match.group(2).strip().strip('"\'')

            observation = self._execute_tool(tool_name, raw_arg)
            trajectory.append(f"Observation: {observation}")

        return "Unable to answer with available evidence."

    def _execute_tool(self, tool_name: str, arg: str) -> str:
        try:
            if tool_name == "search_papers":
                results = self.search(arg)
                return json.dumps([
                    {"paper_id": r.get("paper_id"), "text": r.get("text", "")[:300]}
                    for r in results[:5]
                ], indent=2)
            elif tool_name == "expand_citation":
                return self.expand(arg)
            elif tool_name == "fetch_abstract":
                return self.abstract(arg)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as exc:
            return f"Tool error: {exc}"

    def _llm(self, prompt: str) -> str:
        try:
            resp = self._client.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.error(f"ReAct LLM call failed: {exc}")
            return "Thought: I encountered an error.\nFinal Answer: Unable to complete due to an error."
