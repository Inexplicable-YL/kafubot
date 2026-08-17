from typing import Any, ClassVar

from kafubot.exceptions import AdapterException

__all__ = [
    "ActionFailed",
    "ApiNotAvailable",
    "ApiTimeout",
    "CQHTTPError",
    "CQHTTPException",
    "NetworkError",
]


class CQHTTPException(AdapterException):  # noqa: N818 - compatibility name
    pass


CQHTTPError = CQHTTPException


class NetworkError(CQHTTPException):
    pass


class ActionFailed(CQHTTPException):  # noqa: N818 - OneBot public exception name
    def __init__(
        self,
        resp: dict[str, Any] | None = None,
        *,
        response: dict[str, Any] | None = None,
    ) -> None:
        resolved = resp if resp is not None else response
        if resolved is None:
            resolved = {}
        self.resp = resolved
        self.response = resolved
        super().__init__(f"OneBot action failed: {resolved!r}")


class ApiNotAvailable(ActionFailed):
    ERROR_CODE: ClassVar[int] = 1404


class ApiTimeout(CQHTTPException):  # noqa: N818 - public protocol exception name
    pass
