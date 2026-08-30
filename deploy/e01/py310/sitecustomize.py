# ruff: noqa: B010, UP017, UP035, UP042
"""Python 3.10 compatibility bootstrap for the frozen TARCA contracts."""

from __future__ import annotations

import datetime
import enum
import sys
import typing

import tomli
from typing_extensions import Self

if not hasattr(enum, "StrEnum"):

    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return str(self.value)

    setattr(enum, "StrEnum", StrEnum)

if not hasattr(typing, "Self"):
    setattr(typing, "Self", Self)

if not hasattr(datetime, "UTC"):
    setattr(datetime, "UTC", datetime.timezone.utc)

sys.modules.setdefault("tomllib", tomli)
