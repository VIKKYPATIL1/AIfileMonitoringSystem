from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class OpenAICompatibleClient:
    """Minimal client for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    @classmethod
    def from_env(
        cls,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
    ) -> OpenAICompatibleClient | None:
        resolved_base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        resolved_api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        resolved_model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not resolved_base_url or not resolved_api_key or not resolved_model:
            return None
        return cls(resolved_base_url, resolved_api_key, resolved_model, timeout_seconds, temperature)

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        content = self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, indent=2, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(self._extract_json(content))

    def chat(self, messages: list[dict[str, str]], response_format: dict[str, str] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec: configured API endpoint
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])

    def _extract_json(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Model response did not contain a JSON object")
        return stripped[start : end + 1]


class LLMClient:
    """AI helper for rule-change analysis through an OpenAI-compatible API."""

    def __init__(self, client: OpenAICompatibleClient | None = None):
        self.client = client

    def propose_rule_changes(self, rules: dict[str, Any], failure_summary: dict[str, Any]) -> str:
        if not self.client:
            return "No OpenAI-compatible API configured; review adaptive_suggestions JSON for rule-change candidates."
        system_prompt = (
            "You are a conservative data quality governance agent. Propose rule changes only when the evidence "
            "suggests the source business process changed. Never remove required checks or weaken controls without "
            "explicitly marking the suggestion for human approval. Return JSON with explanation and suggested_diff."
        )
        response = self.client.complete_json(
            system_prompt,
            {"current_rules": rules, "failure_summary": failure_summary},
        )
        return json.dumps(response, indent=2, default=str)
