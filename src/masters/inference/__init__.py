"""Unified inference engine for local and cloud language models."""

from masters.inference.base import BaseInferenceClient
from masters.inference.cloud import OpenAICompatibleClient
from masters.inference.config import InferenceConfig
from masters.inference.exceptions import (
    InferenceAuthError,
    InferenceConnectionError,
    InferenceError,
    InferenceRateLimitError,
    InferenceResponseError,
    InferenceTimeoutError,
)
from masters.inference.factory import get_inference_client
from masters.inference.mock import MockInferenceClient
from masters.inference.ollama import OllamaClient
from masters.inference.response import InferenceResponse

__all__ = [
    "BaseInferenceClient",
    "InferenceAuthError",
    "InferenceConfig",
    "InferenceConnectionError",
    "InferenceError",
    "InferenceRateLimitError",
    "InferenceResponseError",
    "InferenceResponse",
    "InferenceTimeoutError",
    "MockInferenceClient",
    "OllamaClient",
    "OpenAICompatibleClient",
    "get_inference_client",
]
