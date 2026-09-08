"""Client implementation for local models hosted via Ollama."""

import time
from typing import Any

import httpx

from masters.inference.base import BaseInferenceClient
from masters.inference.config import InferenceConfig
from masters.inference.exceptions import (
    InferenceConnectionError,
    InferenceResponseError,
    InferenceTimeoutError,
)
from masters.inference.response import InferenceResponse

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaClient(BaseInferenceClient):
    """Client for local open-weight models executed via Ollama.

    Args:
        base_url: Ollama API server base URL. Defaults to 'http://localhost:11434'.
        client: Optional pre-configured httpx.Client for dependency injection/testing.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _get_client(self, timeout_sec: float) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=timeout_sec)

    def is_available(self) -> bool:
        """Check if the local Ollama daemon is reachable."""
        try:
            with self._get_client(timeout_sec=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        config: InferenceConfig,
    ) -> InferenceResponse:
        base = config.base_url.rstrip("/") if config.base_url else self.base_url
        endpoint = f"{base}/api/chat"

        # Map standard inference config to Ollama runtime options
        options: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        }
        if config.top_k is not None:
            options["top_k"] = config.top_k
        if config.seed is not None:
            options["seed"] = config.seed
        if config.extra_params:
            options.update(config.extra_params)

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        headers = dict(config.extra_headers)
        start_time = time.perf_counter()

        client_ctx = self._get_client(timeout_sec=config.timeout_sec)
        should_close = self._client is None

        try:
            resp = client_ctx.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError as exc:
            raise InferenceConnectionError(
                f"Failed to connect to Ollama at '{self.base_url}'. "
                "Ensure the Ollama daemon is running (`ollama serve`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise InferenceTimeoutError(
                f"Ollama request for model '{config.model}' timed out "
                f"after {config.timeout_sec}s."
            ) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"Ollama HTTP error {exc.response.status_code}: {exc.response.text}"
            if exc.response.status_code == 404:
                msg += (
                    f" (Check if model '{config.model}' is installed via "
                    f"`ollama pull {config.model}`)."
                )
            raise InferenceResponseError(msg) from exc
        except Exception as exc:
            raise InferenceResponseError(
                f"Unexpected error communicating with Ollama: {exc}"
            ) from exc
        finally:
            if should_close:
                client_ctx.close()

        elapsed = time.perf_counter() - start_time

        # Extract generated content and execution metrics
        message = data.get("message", {})
        content = message.get("content", "")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        total_duration_ns = data.get("total_duration")
        reported_latency = (
            total_duration_ns / 1e9 if total_duration_ns is not None else elapsed
        )

        return InferenceResponse(
            content=content,
            model=data.get("model", config.model),
            latency_sec=reported_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=data.get("done_reason"),
            raw_metadata=data,
        )
