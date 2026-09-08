"""Response data structure for inference outputs."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceResponse:
    """Standardized output telemetry from an LLM inference call.

    Attributes:
        content: Generated response text or code.
        model: Exact model identifier reported by provider.
        latency_sec: Total round-trip execution latency in seconds.
        prompt_tokens: Number of prompt/input tokens processed.
        completion_tokens: Number of output/completion tokens generated.
        total_tokens: Total token count (prompt + completion).
        finish_reason: Stop reason reported by provider ('stop', 'length', etc.).
        system_fingerprint: Backend configuration hash (crucial for API drift tracking).
        raw_metadata: Complete raw response dictionary from the provider.
        created_at: Unix epoch timestamp when the response was constructed.
    """

    content: str
    model: str
    latency_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
