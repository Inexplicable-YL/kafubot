from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Generic, Self, TypeVar, cast
from typing_extensions import override

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

MessageT = TypeVar("MessageT", bound="Message[Any]")
MessageSegmentT = TypeVar("MessageSegmentT", bound="MessageSegment[Any]")
BuildMessageType = Iterable[MessageSegmentT] | MessageSegmentT | str | Mapping[str, Any]


class Message(ABC, list[MessageSegmentT], Generic[MessageSegmentT]):
    """Small protocol-neutral message container used by adapters."""

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


class MessageSegment(ABC, BaseModel, Generic[MessageT]):
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
        return cls.model_validate(msg)

    @override
    def __str__(self) -> str:
        return str(self.data)

    @override
    def __repr__(self) -> str:
        return f"MessageSegment<{self.type}>:{self!s}"

    @override
    def __hash__(self) -> int:
        return hash((self.type, tuple(sorted(self.data.items()))))

    def __add__(self, other: Any) -> MessageT:
        return self.get_message_class()(self) + cast("Any", other)

    def __radd__(self, other: Any) -> MessageT:
        return self.get_message_class()(cast("Any", other)) + self

    def is_text(self) -> bool:
        return self.type == "text"
