"""Abstract base interface for LLM inference clients."""

from abc import ABC, abstractmethod

from masters.inference.config import InferenceConfig
from masters.inference.response import InferenceResponse


class BaseInferenceClient(ABC):
    """Abstract interface defining standard interactions with LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the inference provider (e.g., 'ollama', 'openai', 'mock')."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        config: InferenceConfig,
    ) -> InferenceResponse:
        """Send a structured message history and return the model's response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            config: Inference configuration hyperparameters.

        Returns:
            InferenceResponse containing generated text and execution metadata.
        """

    def generate(
        self,
        prompt: str,
        config: InferenceConfig,
    ) -> InferenceResponse:
        """Send a single prompt string and return the model's response.

        Default implementation wraps `prompt` into a single user message.

        Args:
            prompt: Text prompt string.
            config: Inference configuration hyperparameters.

        Returns:
            InferenceResponse containing generated text and execution metadata.
        """
        return self.chat(messages=[{"role": "user", "content": prompt}], config=config)

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider backend or service is reachable."""
