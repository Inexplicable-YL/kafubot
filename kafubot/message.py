from __future__ import annotations

# pyright: reportIncompatibleMethodOverride=false, reportOverlappingOverload=false
from abc import ABC, abstractmethod
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import Any, Generic, Literal, Self, SupportsIndex, TypeVar, cast, overload
from typing_extensions import override

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

__all__ = [
    "BuildMessageType",
    "Message",
    "MessageSegment",
    "MessageSegmentT",
    "MessageT",
]

MessageT = TypeVar("MessageT", bound="Message[Any]")
MessageSegmentT = TypeVar("MessageSegmentT", bound="MessageSegment[Any]")
BuildMessageType = Iterable[MessageSegmentT] | MessageSegmentT | str | Mapping[str, Any]


class Message(ABC, list[MessageSegmentT], Generic[MessageSegmentT]):
    """Small protocol-neutral message container used by adapters."""

    __hash__ = None

    def __init__(self, *messages: BuildMessageType[MessageSegmentT]) -> None:
        segment_class = self.get_segment_class()
        for message in messages:
            if isinstance(message, segment_class):
                self.append(message)
            elif isinstance(message, str):
                self.append(segment_class.from_str(message))
            elif isinstance(message, Mapping):
                self.append(
                    segment_class.from_mapping(cast("Mapping[str, Any]", message))
                )
            elif isinstance(message, Iterable):
                self.extend(cast("Iterable[MessageSegmentT]", message))
            else:
                raise TypeError(f"unsupported message value: {type(message)!r}")

    @classmethod
    @abstractmethod
    def get_segment_class(cls) -> type[MessageSegmentT]: ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source: type[Any],
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.no_info_after_validator_function(
                    cls,
                    handler.generate_schema(list[cls.get_segment_class()]),  # type: ignore[misc]
                ),
            ]
        )

    @override
    def __repr__(self) -> str:
        return f"Message:[{','.join(map(repr, self))}]"

    @override
    def __str__(self) -> str:
        return "".join(map(str, self))

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, Iterable | MessageSegment | Mapping):
            return super().__eq__(
                self.__class__(cast("BuildMessageType[MessageSegmentT]", other))
            )
        return False

    @override
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @override
    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in str(self)
        return super().__contains__(item)

    @override
    def __add__(  # type: ignore[override]
        self,
        other: BuildMessageType[MessageSegmentT],
    ) -> Self:
        return self.__class__(self).__iadd__(other)

    def __radd__(self, other: BuildMessageType[MessageSegmentT]) -> Self:
        return self.__class__(other).__iadd__(self)

    @override
    def __iadd__(  # type: ignore[override]
        self,
        other: BuildMessageType[MessageSegmentT],
    ) -> Self:
        try:
            self.extend(self.__class__(other))
        except TypeError as exc:
            raise TypeError(
                f"unsupported operand types for +: {type(self)!r} and {type(other)!r}"
            ) from exc
        return self

    def is_text(self) -> bool:
        return all(segment.is_text() for segment in self)

    def get_plain_text(self) -> str:
        return "".join(str(segment) for segment in self if segment.is_text())

    def filter_message(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> Self:
        return self.__class__(
            segment
            for segment in self
            if (not include or segment.type in include)
            and (not exclude or segment.type not in exclude)
        )

    @override
    def copy(self) -> Self:
        return self.__class__(self)

    @overload
    def startswith(
        self,
        prefix: str | MessageSegmentT | tuple[str | MessageSegmentT, ...],
        start: SupportsIndex | None = ...,
        end: SupportsIndex | None = ...,
        ignorecase: bool = ...,
        *,
        return_key: Literal[False] = ...,
    ) -> bool: ...

    @overload
    def startswith(
        self,
        prefix: str | MessageSegmentT | tuple[str | MessageSegmentT, ...],
        start: SupportsIndex | None = ...,
        end: SupportsIndex | None = ...,
        ignorecase: bool = ...,
        *,
        return_key: Literal[True] = ...,
    ) -> str | MessageSegmentT | None: ...

    def startswith(
        self,
        prefix: str | MessageSegmentT | tuple[str | MessageSegmentT, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
        ignorecase: bool = False,
        *,
        return_key: bool = False,
        default: str | MessageSegmentT | None = None,
    ) -> bool | str | MessageSegmentT | None:
        if isinstance(prefix, str):
            text = str(self).casefold() if ignorecase else str(self)
            candidate = prefix.casefold() if ignorecase else prefix
            if text.startswith(candidate, start, end):
                return candidate if return_key else True
        elif isinstance(prefix, self.get_segment_class()):
            if self and self[0] == prefix:
                return prefix if return_key else True
        elif isinstance(prefix, tuple):
            text = str(self).casefold() if ignorecase else str(self)
            first = self[0] if self else None
            for item in prefix:
                if isinstance(item, str):
                    candidate = item.casefold() if ignorecase else item
                    if text.startswith(candidate, start, end):
                        return item if return_key else True
                elif isinstance(item, self.get_segment_class()):
                    if first == item:
                        return item if return_key else True
                else:
                    raise TypeError(
                        f"prefix must be str or {self.get_segment_class().__name__}"
                    )
        else:
            raise TypeError(
                f"prefix must be str or {self.get_segment_class().__name__}, "
                f"not {type(prefix).__name__}"
            )
        return default if return_key else False

    @overload
    def endswith(
        self,
        suffix: str | MessageSegmentT | tuple[str | MessageSegmentT, ...],
        start: SupportsIndex | None = ...,
        end: SupportsIndex | None = ...,
        ignorecase: bool = ...,
        *,
        return_key: Literal[False] = ...,
    ) -> bool: ...

    @overload
    def endswith(
        self,
        suffix: str | MessageSegmentT | tuple[str | MessageSegmentT, ...],
        start: SupportsIndex | None = ...,
        end: SupportsIndex | None = ...,
        ignorecase: bool = ...,
        *,
        return_key: Literal[True] = ...,
    ) -> str | MessageSegmentT | None: ...

    def endswith(
        self,
        suffix: str | MessageSegmentT | tuple[str | MessageSegmentT, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
        ignorecase: bool = False,
        *,
        return_key: bool = False,
        default: str | MessageSegmentT | None = None,
    ) -> bool | str | MessageSegmentT | None:
        if isinstance(suffix, str):
            text = str(self).casefold() if ignorecase else str(self)
            candidate = suffix.casefold() if ignorecase else suffix
            if text.endswith(candidate, start, end):
                return candidate if return_key else True
        elif isinstance(suffix, self.get_segment_class()):
            if self and self[-1] == suffix:
                return suffix if return_key else True
        elif isinstance(suffix, tuple):
            text = str(self).casefold() if ignorecase else str(self)
            last = self[-1] if self else None
            for item in suffix:
                if isinstance(item, str):
                    candidate = item.casefold() if ignorecase else item
                    if text.endswith(candidate, start, end):
                        return item if return_key else True
                elif isinstance(item, self.get_segment_class()):
                    if last == item:
                        return item if return_key else True
                else:
                    raise TypeError(
                        f"suffix must be str or {self.get_segment_class().__name__}"
                    )
        else:
            raise TypeError(
                f"suffix must be str or {self.get_segment_class().__name__}, "
                f"not {type(suffix).__name__}"
            )
        return default if return_key else False

    @overload
    def replace(self, old: str, new: str, count: int = -1) -> Self: ...

    @overload
    def replace(
        self,
        old: MessageSegmentT,
        new: MessageSegmentT | None,
        count: int = -1,
    ) -> Self: ...

    def replace(
        self,
        old: str | MessageSegmentT,
        new: str | MessageSegmentT | None,
        count: int = -1,
    ) -> Self:
        if isinstance(old, str):
            if not isinstance(new, str):
                raise TypeError("a string replacement needs a string replacement value")
            return self._replace_str(old, new, count)
        if isinstance(old, self.get_segment_class()):
            if not (isinstance(new, self.get_segment_class()) or new is None):
                raise TypeError(
                    "a message-segment replacement needs a segment or None value"
                )
            result = self.__class__()
            remaining = count
            for item in self:
                if remaining != 0 and item == old:
                    if remaining > 0:
                        remaining -= 1
                    if new is not None:
                        result.append(new)
                else:
                    result.append(item)
            return result
        raise TypeError("old must be str or a message segment")

    def _replace_str(self, old: str, new: str, count: int = -1) -> Self:
        result = self.__class__(*(segment.model_copy(deep=True) for segment in self))
        remaining = count
        for segment in result:
            if remaining == 0:
                break
            if segment.is_text() and old in str(segment.data.get("text", "")):
                text = str(segment.data.get("text", ""))
                if remaining == -1:
                    segment.data["text"] = text.replace(old, new)
                else:
                    replacements = min(remaining, text.count(old))
                    segment.data["text"] = text.replace(old, new, replacements)
                    remaining -= replacements
        return result


class MessageSegment(ABC, BaseModel, Mapping[str, Any], Generic[MessageT]):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    @abstractmethod
    def get_message_class(cls) -> type[MessageT]: ...

    @classmethod
    @abstractmethod
    def from_str(cls, msg: str) -> Self: ...

    @classmethod
    def from_mapping(cls, msg: Mapping[str, Any]) -> Self:
        return cls(**msg)

    @override
    def __str__(self) -> str:
        return str(self.data)

    @override
    def __repr__(self) -> str:
        return f"MessageSegment<{self.type}>:{self!s}"

    @override
    def __hash__(self) -> int:
        return hash((self.type, tuple(sorted(self.data.items()))))

    @override
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    @override
    def __len__(self) -> int:
        return len(self.data)

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self.data

    @override
    def __contains__(self, key: object) -> bool:
        return key in self.data

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.type == other.type
            and self.data == other.data
        )

    @override
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __add__(self, other: Any) -> MessageT:
        return self.get_message_class()(self) + cast("Any", other)

    def __radd__(self, other: Any) -> MessageT:
        return self.get_message_class()(cast("Any", other)) + self

    @override
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @override
    def keys(self) -> KeysView[str]:
        return self.data.keys()

    @override
    def values(self) -> ValuesView[Any]:
        return self.data.values()

    @override
    def items(self) -> ItemsView[str, Any]:
        return self.data.items()

    def is_text(self) -> bool:
        return self.type == "text"
