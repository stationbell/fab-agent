"""Explicit registry for tested deterministic design pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fab_agent.application.pipeline import execute_pipeline
from fab_agent.config import AppConfig
from fab_agent.domain.results import FabResult
from fab_agent.errors import RunStateError
from fab_agent.infrastructure.catalogs import CatalogBundle
from fab_agent.ports import Clock, ModelClient, RunStore


class PipelineExecutor(Protocol):
    async def __call__(
        self,
        *,
        run_id: str,
        store: RunStore,
        model: ModelClient,
        config: AppConfig,
        catalogs: CatalogBundle,
        clock: Clock,
        extract_image: bool,
    ) -> FabResult: ...


@dataclass(frozen=True, slots=True)
class DesignHandler:
    design_type: str
    execute: PipelineExecutor


_HANDLERS = {
    "pipe_spool": DesignHandler(
        design_type="pipe_spool",
        execute=execute_pipeline,
    )
}


def get_design_handler(design_type: str) -> DesignHandler:
    normalized = design_type.strip().lower().replace("-", "_")
    handler = _HANDLERS.get(normalized)
    if handler is None:
        supported = ", ".join(sorted(_HANDLERS))
        raise RunStateError(
            f"Unsupported design type {design_type!r}; supported types: {supported}"
        )
    return handler
