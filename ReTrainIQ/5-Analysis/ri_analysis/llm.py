"""llm.py for ri_analysis"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

IS_STUB = True

@runtime_checkable
class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Produce structured JSON output matching schema."""
        ...

    def complete_text(self, system: str, user: str) -> str:
        """Produce unstructured text response."""
        ...

class MockLLM:
    """Mock LLM client returning configurable dict or text."""
    def __init__(self, json_response: dict | None = None, text_response: str = ""):
        self.json_response = json_response or {}
        self.text_response = text_response
        self.calls: list[dict] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.json_response

    def complete_text(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.text_response

def call_llm(prompt: str, model: str = "mock") -> str:
    return "Stub LLM response."
