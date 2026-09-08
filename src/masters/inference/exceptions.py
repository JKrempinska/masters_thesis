"""Exceptions raised by the masters.inference module."""


class InferenceError(Exception):
    """Base exception for all inference errors."""


class InferenceConnectionError(InferenceError):
    """Raised when connecting to the model server or API fails."""


class InferenceTimeoutError(InferenceError):
    """Raised when an inference call exceeds its configured timeout."""


class InferenceRateLimitError(InferenceError):
    """Raised when an API rate limit (HTTP 429) is encountered."""


class InferenceAuthError(InferenceError):
    """Raised when authentication with an API provider fails (HTTP 401/403)."""


class InferenceResponseError(InferenceError):
    """Raised when an inference endpoint returns an invalid or error response."""
