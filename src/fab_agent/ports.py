"""Narrow interfaces at application boundaries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.provenance import ProvenanceDocument
from fab_agent.domain.takeoff import Takeoff
from fab_agent.domain.validation import ValidationReport

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ModelClient(Protocol):
    @property
    def vision_model(self) -> str: ...

    async def extract_image(
        self,
        *,
        prompt: str,
        response_type: type[ResponseT],
        image_path: Path,
    ) -> ResponseT: ...

    async def health(self) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class RunStore(Protocol):
    root: Path

    def create_run(
        self,
        *,
        source_image: Path,
        source_extension: str,
        normalized_jpeg: bytes,
        source_type: str,
        source_reference: str | None,
        metadata: dict[str, str],
        demo_mode: bool,
    ) -> tuple[str, Path]: ...

    def save_draft(
        self, run_id: str, design: FabricationDesign, provenance: ProvenanceDocument
    ) -> None: ...

    def load_draft(self, run_id: str) -> tuple[FabricationDesign, ProvenanceDocument]: ...

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None: ...

    def normalized_image_path(self, run_id: str) -> Path: ...

    def current(self, run_id: str) -> dict[str, Any]: ...

    def run_metadata(self, run_id: str) -> dict[str, Any]: ...

    def require_awaiting_input(self, run_id: str) -> dict[str, Any]: ...

    def set_status(
        self,
        run_id: str,
        status: Literal["processing", "complete", "awaiting_input", "needs_review"],
    ) -> None: ...

    def commit_version(
        self,
        run_id: str,
        *,
        status: Literal["complete", "awaiting_input", "needs_review"],
        design: FabricationDesign,
        provenance: ProvenanceDocument,
        validation: ValidationReport,
        takeoff: Takeoff | None,
        artifacts_source: Path | None,
        question: str | None = None,
        target_field: str | None = None,
        diagnostics: list[str] | None = None,
    ) -> tuple[int, Path]: ...

    def active_version_path(self, run_id: str) -> Path: ...

    def load_active_design(self, run_id: str) -> FabricationDesign: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    def new_id(self) -> str: ...
