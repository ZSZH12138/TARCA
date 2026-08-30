from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from tarca.contracts import StrictContractModel

E01Condition = Literal[
    "CORRECT_SCM",
    "WRONG_SCM",
    "WRONG_LAG",
    "RANDOM_CONCEPT",
    "IDENTITY",
]


class E01RuntimeProfile(StrictContractModel):
    """Server contract shared by the active E01-v2 runtime."""

    profile_id: str
    base_image: str
    expected_physical_cpu_cores: int = Field(gt=0)
    expected_ram_gib: int = Field(gt=0)
    expected_gpu_count: int = Field(gt=0)
    expected_gpu_name_substring: str
    expected_gpu_vram_gib: int = Field(gt=0)
    minimum_free_storage_gib: int = Field(gt=0)
    reset_limit_hours: int = Field(gt=0)
    reset_margin_hours: int = Field(gt=0)
    host_bind_required: Literal[True]
    capacity_probe_required: Literal[True]
    monitor_bind_host: Literal["127.0.0.1"]

    @field_validator("profile_id", "expected_gpu_name_substring")
    @classmethod
    def _profile_text_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("runtime profile text must be nonblank")
        return value

    @field_validator("base_image")
    @classmethod
    def _base_image_is_exact(cls, value: str) -> str:
        expected = "pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04"
        if value != expected:
            raise ValueError(f"runtime base image must be {expected}")
        return value

    @model_validator(mode="after")
    def _reset_window_has_recovery_margin(self) -> Self:
        if self.reset_margin_hours >= self.reset_limit_hours:
            raise ValueError("reset margin must be shorter than the reset limit")
        return self
