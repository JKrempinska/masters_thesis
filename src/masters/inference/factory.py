"""Factory function for instantiating inference clients."""

from typing import Any

from masters.inference.base import BaseInferenceClient
from masters.inference.cloud import OpenAICompatibleClient
from masters.inference.mock import MockInferenceClient
from masters.inference.ollama import OllamaClient

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def get_inference_client(
    provider: str = "mock",
    **kwargs: Any,
) -> BaseInferenceClient:
    """Instantiate and return an inference client based on the provider name.

    Args:
        provider: Provider identifier ('mock', 'ollama', 'openai', 'gemini',
            'openai_compatible').
        **kwargs: Keyword arguments passed directly to the client constructor.

    Returns:
        Instance of BaseInferenceClient.

    Raises:
        ValueError: If provider is unrecognized.
    """
    normalized = provider.strip().lower()

    if normalized == "mock":
        return MockInferenceClient(**kwargs)
    if normalized == "ollama":
        return OllamaClient(**kwargs)
    if normalized in ("openai", "openai_compatible", "cloud"):
        return OpenAICompatibleClient(**kwargs)
    if normalized == "gemini":
        if "base_url" not in kwargs:
            kwargs["base_url"] = GEMINI_OPENAI_BASE_URL
        return OpenAICompatibleClient(**kwargs)

    raise ValueError(
        f"Unknown inference provider '{provider}'. Supported providers are: "
        "'mock', 'ollama', 'openai', 'gemini', 'openai_compatible'."
    )
