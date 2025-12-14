"""DeepSeek client wrapper used by the StoryRag demo (no OpenAI dependency)."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv
from requests import Response

load_dotenv()

DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "30"))


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API returns an error response."""


class DeepSeekClient:
    """Minimal DeepSeek chat client with env-based configuration."""

    def __init__(
            self,
            *,
            # Optional[str] 表示这个参数的类型注解，说明它可以是字符串类型（str）或者 None
            api_key: Optional[str] = None,
            base_url: str = DEFAULT_BASE_URL,
            model: str = DEFAULT_MODEL,
            timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY. Set it in .env or environment.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat_completion(
            self,
            messages: Iterable[dict],
            *,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            proxies: Optional[dict] = None,
    ) -> str:
        """
        Call DeepSeek chat completions API and return the message content.

        messages should follow OpenAI-compatible schema.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            resp: Response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                proxies=proxies,
            )
        except requests.RequestException as exc:  # noqa: PERF203
            raise DeepSeekError(f"Request failed: {exc}") from exc

        if resp.status_code != 200:
            raise DeepSeekError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if "error" in data:
            raise DeepSeekError(str(data["error"]))

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise DeepSeekError(f"Malformed response: {data}") from exc
