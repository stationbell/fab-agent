"""Trigger-neutral run, resume, and deterministic rebuild use cases."""

from __future__ import annotations

from dataclasses import dataclass

from fab_agent.application.pipeline import PipelineState, commit_terminal
from fab_agent.application.policy import set_human_field
from fab_agent.application.registry import DesignHandler, get_design_handler
from fab_agent.config import AppConfig
from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.dimensions import parse_dimension
from fab_agent.domain.provenance import (
    ProvenanceDocument,
    ProvenanceEntry,
    changed_scalar_fields,
)
from fab_agent.domain.results import FabRequest, FabResult
from fab_agent.domain.takeoff import compute_takeoff
from fab_agent.domain.validation import validate_design
from fab_agent.errors import RunStateError
from fab_agent.infrastructure.artifacts import generate_artifacts
from fab_agent.infrastructure.catalogs import CatalogBundle
from fab_agent.infrastructure.images import normalize_image
from fab_agent.ports import Clock, ModelClient, RunStore


@dataclass(frozen=True, slots=True)
class Dependencies:
    config: AppConfig
    model: ModelClient
    store: RunStore
    catalogs: CatalogBundle
    clock: Clock


def _with_demo_catalogs(config: AppConfig, enabled: bool) -> AppConfig:
    if not enabled or config.catalogs.allow_demo_entries:
        return config
    return config.model_copy(
        update={"catalogs": config.catalogs.model_copy(update={"allow_demo_entries": True})}
    )


def _stored_run_context(
    run_id: str,
    dependencies: Dependencies,
) -> tuple[DesignHandler, AppConfig]:
    run_metadata = dependencies.store.run_metadata(run_id)
    metadata = run_metadata.get("metadata", {})
    design_type = (
        str(metadata.get("design_type", "pipe_spool"))
        if isinstance(metadata, dict)
        else "pipe_spool"
    )
    config = _with_demo_catalogs(
        dependencies.config,
        bool(run_metadata.get("demo_mode", False)),
    )
    return get_design_handler(design_type), config


async def start_run(request: FabRequest, dependencies: Dependencies) -> FabResult:
    handler = get_design_handler(request.design_type)
    config = _with_demo_catalogs(dependencies.config, request.demo)
    normalized = normalize_image(request.input_image, config.images)
    run_id, _ = dependencies.store.create_run(
        source_image=request.input_image,
        source_extension=normalized.source_extension,
        normalized_jpeg=normalized.normalized_jpeg,
        source_type=request.source_type,
        source_reference=request.source_reference,
        metadata={**request.metadata, "design_type": handler.design_type},
        demo_mode=request.demo or config.catalogs.allow_demo_entries,
    )
    dependencies.store.save_draft(run_id, FabricationDesign(), ProvenanceDocument())
    dependencies.store.append_event(
        run_id,
        "image.normalized",
        {
            "format": normalized.source_format,
            "width": normalized.width,
            "height": normalized.height,
        },
    )
    return await handler.execute(
        run_id=run_id,
        store=dependencies.store,
        model=dependencies.model,
        config=config,
        catalogs=dependencies.catalogs,
        clock=dependencies.clock,
        extract_image=True,
    )


async def resume_run(run_id: str, human_answer: str, dependencies: Dependencies) -> FabResult:
    if not human_answer.strip():
        raise RunStateError("Human answer cannot be empty")
    current = dependencies.store.require_awaiting_input(run_id)
    handler, config = _stored_run_context(run_id, dependencies)
    target_field = str(current["target_field"])
    design, provenance = dependencies.store.load_draft(run_id)
    design_data = design.model_dump(mode="python", exclude_none=True)
    set_human_field(design_data, target_field, human_answer.strip())
    design = FabricationDesign.model_validate(design_data)
    provenance.entries.append(
        ProvenanceEntry(
            field_path=target_field,
            source_type="human_answer",
            raw_text=human_answer,
            recorded_at=dependencies.clock.now(),
        )
    )
    dependencies.store.set_status(run_id, "processing")
    dependencies.store.save_draft(run_id, design, provenance)
    dependencies.store.append_event(
        run_id,
        "human_answer.recorded",
        {"target_field": target_field},
    )
    return await handler.execute(
        run_id=run_id,
        store=dependencies.store,
        model=dependencies.model,
        config=config,
        catalogs=dependencies.catalogs,
        clock=dependencies.clock,
        extract_image=False,
    )


async def rebuild_run(run_id: str, dependencies: Dependencies) -> FabResult:
    _, config = _stored_run_context(run_id, dependencies)
    design, provenance = dependencies.store.load_draft(run_id)
    try:
        previous_design = dependencies.store.load_active_design(run_id)
    except RunStateError:
        previous_design = FabricationDesign()
    changes = changed_scalar_fields(
        previous_design.model_dump(mode="python", exclude_none=True),
        design.model_dump(mode="python", exclude_none=True),
    )
    timestamp = dependencies.clock.now()
    provenance.entries.extend(
        ProvenanceEntry(
            field_path=path,
            source_type="manual_edit",
            raw_text=value,
            recorded_at=timestamp,
        )
        for path, value in changes.items()
    )
    dependencies.store.save_draft(run_id, design, provenance)
    validation = validate_design(
        design,
        tolerance=parse_dimension(config.validation.dimensional_tolerance_in).inches,
        catalogs=dependencies.catalogs,
        allow_demo=config.catalogs.allow_demo_entries,
    )
    state = PipelineState(design=design, provenance=provenance, validation=validation)
    if validation.passed:
        state.takeoff = compute_takeoff(design, validation)
        state.artifacts = generate_artifacts(
            design,
            validation,
            state.takeoff,
            dependencies.catalogs,
            allow_demo=config.catalogs.allow_demo_entries,
            cad_enabled=config.cad.enabled,
        )
        return commit_terminal(
            run_id=run_id,
            store=dependencies.store,
            state=state,
            status="complete",
        )
    return commit_terminal(
        run_id=run_id,
        store=dependencies.store,
        state=state,
        status="needs_review",
    )
