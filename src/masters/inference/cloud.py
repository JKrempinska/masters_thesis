"""Client implementation for OpenAI-compatible Cloud APIs (OpenAI, Gemini, etc.)."""

import os
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from masters.inference.base import BaseInferenceClient
from masters.inference.config import InferenceConfig
from masters.inference.exceptions import (
    InferenceAuthError,
    InferenceConnectionError,
    InferenceError,
    InferenceRateLimitError,
    InferenceResponseError,
    InferenceTimeoutError,
)
from masters.inference.response import InferenceResponse

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class _RetryableRateLimitError(Exception):
    """Internal transient exception indicating a 429 or 503 requiring retry."""


class OpenAICompatibleClient(BaseInferenceClient):
    """Client for cloud API endpoints adhering to the OpenAI Chat Completions schema.

    Supports OpenAI, Google Gemini (via OpenAI compatibility endpoint), Groq,
    and open-source inference servers (vLLM, Ollama-openai, Together).

    Args:
        base_url: Base endpoint URL. Defaults to 'https://api.openai.com/v1'.
        api_key: Optional API key. If omitted, checks environment variables.
        client: Optional pre-configured httpx.Client for testing / dependency injection.
        max_retries: Number of exponential backoff retry attempts on 429/503.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    def _get_client(self, timeout_sec: float) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=timeout_sec)

    def _resolve_api_key(self, config: InferenceConfig) -> str | None:
        """Resolve API key in order: config override -> constructor -> env vars."""
        if config.api_key:
            return config.api_key
        if self._api_key:
            return self._api_key
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def is_available(self) -> bool:
        """Check if an API key is present and endpoint base is configured."""
        return self._resolve_api_key(InferenceConfig(model="test")) is not None

    def chat(
        self,
        messages: list[dict[str, str]],
        config: InferenceConfig,
    ) -> InferenceResponse:
        base = config.base_url.rstrip("/") if config.base_url else self.base_url
        endpoint = f"{base}/chat/completions"

        api_key = self._resolve_api_key(config)
        headers = {
            "Content-Type": "application/json",
            **config.extra_headers,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.extra_params:
            payload.update(config.extra_params)

        client_ctx = self._get_client(timeout_sec=config.timeout_sec)
        should_close = self._client is None

        # Build retry-wrapped request execution
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
            retry=retry_if_exception_type(_RetryableRateLimitError),
        )
        def _execute_request() -> dict[str, Any]:
            try:
                resp = client_ctx.post(endpoint, json=payload, headers=headers)
                if resp.status_code == 429 or resp.status_code == 503:
                    msg = (
                        f"Rate limit or service unavailable ({resp.status_code}): "
                        f"{resp.text}"
                    )
                    raise _RetryableRateLimitError(msg)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
            except httpx.ConnectError as exc:
                raise InferenceConnectionError(
                    f"Connection failed to endpoint '{endpoint}': {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise InferenceTimeoutError(
                    f"Request for model '{config.model}' timed out "
                    f"after {config.timeout_sec}s."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                text = exc.response.text
                if status in (401, 403):
                    raise InferenceAuthError(
                        f"Authentication failed (HTTP {status}): {text}"
                    ) from exc
                raise InferenceResponseError(
                    f"HTTP error {status} from '{endpoint}': {text}"
                ) from exc

        start_time = time.perf_counter()
        try:
            data = _execute_request()
        except _RetryableRateLimitError as exc:
            raise InferenceRateLimitError(
                f"Rate limit exceeded after {self.max_retries} retries: {exc}"
            ) from exc
        except Exception as exc:
            if isinstance(exc, InferenceError):
                raise
            raise InferenceResponseError(f"API request failed: {exc}") from exc
        finally:
            if should_close:
                client_ctx.close()

        elapsed = time.perf_counter() - start_time

        choices = data.get("choices", [])
        content = ""
        finish_reason = None
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        system_fp = data.get("system_fingerprint")

        return InferenceResponse(
            content=content,
            model=data.get("model", config.model),
            latency_sec=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            system_fingerprint=system_fp,
            raw_metadata=data,
        )
