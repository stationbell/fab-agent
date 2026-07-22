"""Trigger-neutral public request and result models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FabRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_image: Path
    design_type: str = "pipe_spool"
    source_type: str = "cli"
    source_reference: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    demo: bool = False


class FabResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["complete", "awaiting_input", "needs_review"]
    run_path: Path
    question: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Path] = Field(default_factory=dict)
