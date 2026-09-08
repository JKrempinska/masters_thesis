"""Deterministic mock inference client for testing and offline development."""

import time
from collections.abc import Callable

from masters.inference.base import BaseInferenceClient
from masters.inference.config import InferenceConfig
from masters.inference.response import InferenceResponse


def _default_code_generator(prompt: str, config: InferenceConfig) -> str:
    """Generate deterministic dummy Python code based on prompt keywords."""
    if "impute_missing" in prompt:
        return (
            "```python\n"
            "import pandas as pd\n\n"
            "def impute_missing(df: pd.DataFrame) -> pd.DataFrame:\n"
            "    # Deterministic baseline imputer\n"
            "    imputed = df.copy()\n"
            "    for col in imputed.columns:\n"
            "        if pd.api.types.is_numeric_dtype(imputed[col]):\n"
            "            imputed[col] = imputed[col].fillna(imputed[col].median())\n"
            "        else:\n"
            "            mode_val = imputed[col].mode(dropna=True)\n"
            "            fill_val = (\n"
            "                mode_val.iloc[0] if not mode_val.empty else 'unknown'\n"
            "            )\n"
            "            imputed[col] = imputed[col].fillna(fill_val)\n"
            "    return imputed\n"
            "```"
        )
    return f"Mock response for model '{config.model}' with seed {config.seed}"


class MockInferenceClient(BaseInferenceClient):
    """Offline test double for inference calls.

    Generates deterministic responses without making network requests or requiring
    GPU hardware.
    """

    def __init__(
        self,
        canned_responses: list[str] | None = None,
        response_generator: Callable[[str, InferenceConfig], str] | None = None,
        simulate_latency_sec: float = 0.0,
        default_prompt_tokens: int = 100,
        default_completion_tokens: int = 50,
    ) -> None:
        self._canned_responses = list(canned_responses) if canned_responses else []
        self._call_count = 0
        self._response_generator = response_generator or _default_code_generator
        self.simulate_latency_sec = simulate_latency_sec
        self.default_prompt_tokens = default_prompt_tokens
        self.default_completion_tokens = default_completion_tokens

    @property
    def provider_name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def chat(
        self,
        messages: list[dict[str, str]],
        config: InferenceConfig,
    ) -> InferenceResponse:
        start_time = time.perf_counter()
        if self.simulate_latency_sec > 0:
            time.sleep(self.simulate_latency_sec)

        last_prompt = messages[-1]["content"] if messages else ""

        if self._canned_responses:
            idx = self._call_count % len(self._canned_responses)
            content = self._canned_responses[idx]
        else:
            content = self._response_generator(last_prompt, config)

        self._call_count += 1
        elapsed = time.perf_counter() - start_time

        prompt_tokens = self.default_prompt_tokens + len(last_prompt) // 4
        completion_tokens = self.default_completion_tokens + len(content) // 4

        return InferenceResponse(
            content=content,
            model=config.model,
            latency_sec=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            system_fingerprint="mock-fp-0",
            raw_metadata={"mock": True, "call_count": self._call_count},
        )
