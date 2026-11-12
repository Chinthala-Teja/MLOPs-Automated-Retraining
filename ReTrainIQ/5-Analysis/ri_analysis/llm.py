"""llm.py - the only module allowed to make network calls.

The LLM has three jobs: draft the project profile (Module 1), propose a metric
plan (Module 4), and write summaries / chat answers. It never computes a number
and never decides. Everything works without an LLM (get_client() returns None).

Configuration comes from environment variables or a .env file in the repo root:
    LLM_PROVIDER  anthropic | openai | gemini | groq | ollama | <any OpenAI-compatible>
    LLM_API_KEY   the key (not needed for ollama)
    LLM_MODEL     model name
    LLM_BASE_URL  optional override of the API base URL
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ri_init.config import ROOT

IS_STUB = False

# Base URLs of OpenAI-compatible chat endpoints. Check each provider's current
# documentation before relying on one; override with LLM_BASE_URL if it moved.
OPENAI_COMPATIBLE = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}


@runtime_checkable
class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Produce structured JSON output matching schema."""
        ...

    def complete_text(self, system: str, user: str) -> str:
        """Produce unstructured text response."""
        ...


class MockLLM:
    """Mock LLM client returning configurable dict or text (tests, offline demos)."""
    def __init__(self, json_response: dict | None = None, text_response: str | List[str] = ""):
        self.json_response = json_response or {}
        self.text_response = text_response
        self.calls: list[dict] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.json_response

    def complete_text(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if isinstance(self.text_response, list):
            n_text = sum(1 for c in self.calls if "schema" not in c)
            return self.text_response[min(n_text - 1, len(self.text_response) - 1)]
        return self.text_response


class LLMError(Exception):
    """Raised when the LLM call fails or returns something unusable."""


class HTTPLLM:
    """Real client: Anthropic Messages API or any OpenAI-compatible chat endpoint. Temperature 0."""

    def __init__(self, provider: str, api_key: str, model: str, base_url: Optional[str] = None, timeout: int = 60):
        self.provider = provider.lower().strip()
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or OPENAI_COMPATIBLE.get(self.provider, "")).rstrip("/")
        self.timeout = timeout

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests  # imported here so the rest of the system never needs the network

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"LLM returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def complete_text(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            data = self._post(
                (self.base_url or "https://api.anthropic.com") + "/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                {"model": self.model, "max_tokens": 1500, "temperature": 0, "system": system,
                 "messages": [{"role": "user", "content": user}]},
            )
            text = "".join(part.get("text", "") for part in data.get("content", []))
        else:
            if not self.base_url:
                raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'; set LLM_BASE_URL.")
            data = self._post(
                self.base_url + "/chat/completions",
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                {"model": self.model, "temperature": 0,
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            )
            text = data["choices"][0]["message"]["content"]
        try:
            import ri_init.store as store
            store.add_llm_log(self.model, system, user, text)
        except Exception:  # logging must never break a request
            pass
        return text

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        text = self.complete_text(
            system + "\nReply with ONE JSON object only, no prose, no code fences. It must follow this schema:\n"
            + json.dumps(schema),
            user,
        )
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMError("LLM reply contained no JSON object.")
        try:
            return json.loads(match.group(0))
        except ValueError as exc:
            raise LLMError(f"LLM reply was not valid JSON: {exc}") from exc


def _read_dotenv() -> Dict[str, str]:
    path = ROOT / ".env"
    out: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def get_client() -> Optional[LLMClient]:
    """The configured client, or None when no LLM is set up (the system then uses its fallbacks)."""
    env = {**_read_dotenv(), **{k: v for k, v in os.environ.items() if k.startswith("LLM_")}}
    provider, model = env.get("LLM_PROVIDER", "").strip(), env.get("LLM_MODEL", "").strip()
    key = env.get("LLM_API_KEY", "").strip()
    if not provider or not model or (not key and provider.lower() != "ollama"):
        return None
    return HTTPLLM(provider, key or "ollama", model, env.get("LLM_BASE_URL") or None)


def propose_metric_plan(profile: Dict[str, Any], llm: Optional[LLMClient]) -> List[str]:
    """Module 4 metric plan: the LLM may only pick names from the registry; otherwise the default plan."""
    from ri_metrics.metrics import REGISTRY, default_plan, validate_plan

    fallback = default_plan(profile)
    if llm is None:
        return fallback
    try:
        reply = llm.complete_json(
            "You choose evaluation metrics for a deployed ML model. Choose only names from the allowed list.",
            json.dumps({"problem": profile.get("problem_statement"), "task": profile.get("task"),
                        "primary_metric": profile.get("primary_metric"), "allowed": fallback,
                        "descriptions": {n: REGISTRY[n]["description"] for n in fallback}}),
            {"type": "object", "properties": {"metrics": {"type": "array", "items": {"type": "string"}}},
             "required": ["metrics"]},
        )
        plan = validate_plan(reply.get("metrics", []), profile)
    except (LLMError, ValueError, AttributeError, TypeError):
        return fallback
    primary = str(profile.get("primary_metric", "")).lower()
    if primary in fallback and primary not in plan:
        plan.insert(0, primary)
    return plan or fallback
