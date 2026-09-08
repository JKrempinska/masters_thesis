"""Configuration dataclass for inference calls."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    """Hyperparameter and operational configuration for an LLM inference call.

    Attributes:
        model: Model identifier (e.g. 'qwen2.5-coder:7b', 'gpt-4o-mini').
        temperature: Sampling temperature in [0.0, 2.0]. Higher values yield more
            stochastic responses.
        top_p: Nucleus sampling probability threshold in (0.0, 1.0].
        top_k: Top-k sampling truncation integer, or None if disabled.
        seed: Random seed for deterministic generation if supported by runtime.
        max_tokens: Maximum number of tokens to generate.
        timeout_sec: Maximum request execution time in seconds.
        base_url: Optional endpoint URL override (e.g., custom Ollama or proxy host).
        api_key: Optional API authentication key (falls back to provider env var).
        extra_headers: Optional additional HTTP headers.
        extra_params: Optional provider-specific parameter dictionary.
    """

    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int = 2048
    timeout_sec: float = 60.0
    base_url: str | None = None
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be strictly in (0.0, 1.0], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(
                f"top_k must be a positive integer or None, got {self.top_k}"
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be greater than or equal to 1, got {self.max_tokens}"
            )
        if self.timeout_sec <= 0.0:
            raise ValueError(f"timeout_sec must be positive, got {self.timeout_sec}")
