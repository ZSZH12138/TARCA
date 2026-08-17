"""Architecture-level constants that do not change scientific semantics."""

from __future__ import annotations

from enum import StrEnum

ARCHITECTURE_VERSION = "1.0"


class ArchitecturePlane(StrEnum):
    """The three independent TARCA responsibility planes."""

    SCIENCE = "science"
    GOVERNANCE = "governance"
    EXECUTION = "execution"
