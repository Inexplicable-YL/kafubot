"""Custom exceptions for the Dify client."""

from typing import Any


class DifyClientError(Exception):
    """Base exception for all Dify client errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response


class APIError(DifyClientError):
    """Raised when the API returns an error response."""

    def __init__(
        self, message: str, status_code: int, response: dict[str, Any] | None = None
    ):
        super().__init__(message, status_code, response)
        self.status_code = status_code


class AuthenticationError(DifyClientError):
    """Raised when authentication fails."""


class RateLimitError(DifyClientError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: int | None = None
    ):
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(DifyClientError):
    """Raised when request validation fails."""


class NetworkError(DifyClientError):
    """Raised when network-related errors occur."""


class DifyTimeoutError(DifyClientError):
    """Raised when request times out."""


class FileUploadError(DifyClientError):
    """Raised when file upload fails."""


class DatasetError(DifyClientError):
    """Raised when dataset operations fail."""


class WorkflowError(DifyClientError):
    """Raised when workflow operations fail."""
