"""Utility wrapper for interacting with Anthropic's Claude models.

This module centralises configuration and error handling so the rest of the
codebase can request generations without depending directly on the external
SDK. The implementation is deliberately lightweight to remain compatible with
restricted execution environments. When no API key is configured we raise a
custom error so callers can gracefully fall back to heuristic logic.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)


class LLMGenerationError(RuntimeError):
    """Raised when the language model could not produce a response."""


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


class ClaudeClient:
    """Minimal client for Anthropic's Claude Messages API."""

    api_url = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "claude-3-5-sonnet-20240620",
        timeout: float = 30.0,
        max_output_tokens: int = 1024,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", model)
        self.max_output_tokens = max_output_tokens
        self._client = client or httpx.Client(timeout=timeout, headers={"User-Agent": "LexiAI/0.4"})

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        messages: Sequence[LLMMessage | dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        if not self.is_configured:
            raise LLMGenerationError("Claude API key not configured")

        serialised_messages = [self._serialise_message(message) for message in messages]
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_output_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": serialised_messages,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = self._client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            raise LLMGenerationError(f"Claude API request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise LLMGenerationError("Claude API response was not valid JSON") from exc

        content_blocks = data.get("content", [])
        combined = "".join(
            block.get("text", "") for block in content_blocks if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not combined:
            raise LLMGenerationError("Claude API returned an empty response")
        return combined

    def _serialise_message(self, message: LLMMessage | dict[str, str]) -> dict[str, object]:
        if isinstance(message, LLMMessage):
            role = message.role
            content = message.content
        else:
            role = message.get("role", "user")
            content = message.get("content", "")
        return {"role": role, "content": [{"type": "text", "text": content}]}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to close Claude HTTP client", exc_info=True)


__all__ = ["ClaudeClient", "LLMGenerationError", "LLMMessage"]
