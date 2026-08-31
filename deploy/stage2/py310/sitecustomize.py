"""Narrow Python 3.10 compatibility shims for code authored against Python 3.11."""

# ruff: noqa: I001, UP017, UP042

import datetime
import enum
import typing


if not hasattr(enum, "StrEnum"):

    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return str.__str__(self)

    enum.StrEnum = StrEnum  # type: ignore[attr-defined]

if not hasattr(typing, "Self"):
    typing.Self = typing.TypeVar("Self")  # type: ignore[attr-defined]

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
