"""Deterministic pipe-spool workflow with one bounded image extraction."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fab_agent.config import AppConfig
from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.dimensions import parse_dimension
from fab_agent.domain.provenance import (
    ProvenanceDocument,
    ProvenanceEntry,
    observed_source_fields,
)
from fab_agent.domain.results import FabResult
from fab_agent.domain.takeoff import Takeoff, compute_takeoff
from fab_agent.domain.validation import ValidationReport, validate_design
from fab_agent.errors import ArtifactError, ModelError
from fab_agent.infrastructure.artifacts import ArtifactSet, generate_artifacts
from fab_agent.infrastructure.catalogs import CatalogBundle
from fab_agent.models.pipe_spool import convert_observation
from fab_agent.models.schemas import PipeSpoolObservation
from fab_agent.ports import Clock, ModelClient, RunStore


@dataclass(frozen=True, slots=True)
class Clarification:
    question: str
    target_field: str
    reason: str


@dataclass(slots=True)
class PipelineState:
    design: FabricationDesign
    provenance: ProvenanceDocument
    uncertainties: list[str] = field(default_factory=list)
    validation: ValidationReport | None = None
    takeoff: Takeoff | None = None
    artifacts: ArtifactSet | None = None


def vision_prompt() -> str:
    return (
        "Observe this photographed hand sketch once using the supplied flat pipe-spool schema. "
        "Preserve every visible source value as raw text; do not calculate dimensions, supply "
        "missing material, or call the word 'pipe' a material. There may be multiple independent "
        "straight spools. List only physical features in left-to-right order: the two endpoints "
        "and any outlets, couplings, or caps between them. Never list dimension ticks or marks as "
        "physical features. List the dimension strings between those physical features separately "
        "in the same left-to-right order. Python owns all feature IDs and segment links. "
        "Use kind start for the left boundary and end for the right boundary unless a coupling, "
        "cap, or open end is explicitly drawn there. Preserve boundary marks such as G or P in "
        "label and connection_type rather than changing the boundary kind to unknown_end. "
        "Preserve feet and inch marks exactly: for example, 4'-6 1/4\" must not become "
        "4-6 1/4\". Put 'threaded' or 'grooved' in an outlet's connection_type, never in "
        "orientation_raw. When a physical outlet stem visibly points away from the main run, set "
        "orientation to up, right, down, or left from that geometry; do not infer orientation from "
        "text placement alone. Capture explicit orientation legends. When outlet label A or B "
        "matches a legend, copy the label to orientation_raw and copy the legend direction to "
        "orientation. "
        "Return only the measurement text in stated_total_length_raw; exclude field labels such as "
        "Total. Capture project references, notes, and observed parts separately. For every "
        "observed parts-list line, populate kind and nominal_size_raw whenever those words and "
        "values are explicitly visible. End labels such as G "
        "or P and codes such as 005 are source text, not material. Put uncertain readings in "
        "uncertainties rather than inventing a value. Always return spools and "
        "observed_components as JSON arrays, including when either contains one item."
    )


def _error_chain(error: BaseException) -> list[str]:
    """Record exception types without persisting messages or machine-specific paths.

    An unexpected exception type here means a defect rather than an unreadable
    sketch, so the type is recorded even though the operator-facing warning
    stays a plain message.
    """

    chain: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return chain


def _missing_field_clarification(report: ValidationReport) -> Clarification | None:
    fields = {
        "pipe.size_missing": ("nominal_size_raw", "nominal pipe size"),
        "pipe.schedule_missing": ("schedule_raw", "pipe schedule"),
        "pipe.material_missing": ("material_raw", "pipe material"),
        "pipe.total_missing": ("stated_total_length_raw", "total finished length"),
    }
    for issue in report.issues:
        mapping = fields.get(issue.code)
        if mapping is None or issue.spool_id is None:
            continue
        field_name, label = mapping
        spool_label = (
            issue.spool_id
            if issue.spool_id.lower().startswith("spool")
            else f"spool {issue.spool_id}"
        )
        return Clarification(
            question=f"What is the {label} for {spool_label}?",
            target_field=f"spools.{issue.spool_id}.{field_name}",
            reason=issue.message,
        )
    return None


def choose_clarification(state: PipelineState) -> Clarification | None:
    if state.validation is not None:
        return _missing_field_clarification(state.validation)
    return None


def commit_terminal(
    *,
    run_id: str,
    store: RunStore,
    state: PipelineState,
    status: Literal["complete", "awaiting_input", "needs_review"],
    question: str | None = None,
    target_field: str | None = None,
) -> FabResult:
    validation = state.validation or ValidationReport(passed=False)
    try:
        version, version_path = store.commit_version(
            run_id,
            status=status,
            design=state.design,
            provenance=state.provenance,
            validation=validation,
            takeoff=state.takeoff,
            artifacts_source=state.artifacts.root if state.artifacts else None,
            question=question,
            target_field=target_field,
            diagnostics=state.uncertainties,
        )
        store.append_event(run_id, f"run.{status}", {"version": version})
        artifacts: dict[str, Path] = {}
        if state.artifacts:
            for key, source in state.artifacts.files.items():
                artifacts[key] = (
                    version_path / "artifacts" / source.relative_to(state.artifacts.root)
                )
    finally:
        # The version snapshot owns its own copy, so the staging tree is always
        # discarded, including when the commit itself failed.
        if state.artifacts:
            shutil.rmtree(state.artifacts.root, ignore_errors=True)
    warnings = [issue.message for issue in validation.issues if issue.level != "info"]
    warnings.extend(item for item in state.uncertainties if item not in warnings)
    if state.takeoff:
        warnings.extend(state.takeoff.reconciliation_warnings)
    return FabResult(
        run_id=run_id,
        status=status,
        run_path=store.root / run_id,
        question=question,
        warnings=warnings,
        artifacts=artifacts,
    )


async def _extract_once(
    *,
    run_id: str,
    store: RunStore,
    model: ModelClient,
    config: AppConfig,
    clock: Clock,
    state: PipelineState,
) -> None:
    try:
        async with asyncio.timeout(config.ollama.request_timeout_seconds):
            observation = await model.extract_image(
                prompt=vision_prompt(),
                response_type=PipeSpoolObservation,
                image_path=store.normalized_image_path(run_id),
            )
    except TimeoutError as exc:
        raise ModelError(
            "Timed out waiting for image transcription after "
            f"{config.ollama.request_timeout_seconds} seconds"
        ) from exc

    converted = convert_observation(observation)
    state.design = converted.design
    state.uncertainties = list(converted.uncertainties)
    timestamp = clock.now()
    recorded_entries: list[ProvenanceEntry] = []
    for path, raw_text in observed_source_fields(state.design).items():
        recorded_entries.append(
            ProvenanceEntry(
                field_path=path,
                source_type="image",
                raw_text=raw_text,
                recorded_at=timestamp,
                model=model.vision_model,
            )
        )
    state.provenance.entries.extend(recorded_entries)
    store.save_draft(run_id, state.design, state.provenance)
    store.append_event(
        run_id,
        "pipeline.extraction_completed",
        {"model": model.vision_model, "source_fields": len(recorded_entries)},
    )


def _validate(
    *,
    state: PipelineState,
    config: AppConfig,
    catalogs: CatalogBundle,
) -> None:
    tolerance = parse_dimension(config.validation.dimensional_tolerance_in).inches
    state.validation = validate_design(
        state.design,
        tolerance=tolerance,
        catalogs=catalogs,
        allow_demo=config.catalogs.allow_demo_entries,
    )


def _generate(
    *,
    state: PipelineState,
    config: AppConfig,
    catalogs: CatalogBundle,
) -> None:
    assert state.validation is not None and state.validation.passed
    state.takeoff = compute_takeoff(state.design, state.validation)
    state.artifacts = generate_artifacts(
        state.design,
        state.validation,
        state.takeoff,
        catalogs,
        allow_demo=config.catalogs.allow_demo_entries,
        cad_enabled=config.cad.enabled,
    )


def _finalization_error(state: PipelineState, config: AppConfig) -> str | None:
    if state.validation is None or not state.validation.passed:
        return "Finalization requires passing deterministic validation"
    if state.takeoff is None or state.artifacts is None:
        return "Finalization requires takeoff and artifacts"
    required = {"bom", "review"}
    if config.cad.enabled:
        required.update(f"{spool.id}_step" for spool in state.design.spools)
    if not required <= state.artifacts.files.keys():
        return "Finalization artifact gate failed"
    return None


async def execute_pipeline(
    *,
    run_id: str,
    store: RunStore,
    model: ModelClient,
    config: AppConfig,
    catalogs: CatalogBundle,
    clock: Clock,
    extract_image: bool,
) -> FabResult:
    """Run the fixed workflow; only initial extraction is model-authored."""

    design, provenance = store.load_draft(run_id)
    state = PipelineState(design=design, provenance=provenance)
    try:
        if extract_image:
            await _extract_once(
                run_id=run_id,
                store=store,
                model=model,
                config=config,
                clock=clock,
                state=state,
            )
        _validate(state=state, config=config, catalogs=catalogs)
        assert state.validation is not None
        store.append_event(
            run_id,
            "pipeline.validation_completed",
            {
                "passed": state.validation.passed,
                "issues": len(state.validation.issues),
            },
        )

        clarification = choose_clarification(state)
        current = store.current(run_id)
        if (
            clarification is not None
            and int(current["questions_asked"]) < config.workflow.max_human_questions
        ):
            store.append_event(
                run_id,
                "pipeline.clarification_requested",
                {
                    "target_field": clarification.target_field,
                    "reason": clarification.reason,
                },
            )
            return commit_terminal(
                run_id=run_id,
                store=store,
                state=state,
                status="awaiting_input",
                question=clarification.question,
                target_field=clarification.target_field,
            )

        if state.validation.passed:
            _generate(state=state, config=config, catalogs=catalogs)

        gate_error = _finalization_error(state, config)
        if gate_error is not None:
            if gate_error not in state.uncertainties and state.validation.passed:
                state.uncertainties.append(gate_error)
            return commit_terminal(
                run_id=run_id,
                store=store,
                state=state,
                status="needs_review",
            )

        assert state.artifacts is not None
        store.append_event(
            run_id,
            "pipeline.outputs_generated",
            {"artifacts": len(state.artifacts.files)},
        )
        return commit_terminal(
            run_id=run_id,
            store=store,
            state=state,
            status="complete",
        )
    except (ArtifactError, ModelError, ValueError) as exc:
        store.append_event(
            run_id,
            "pipeline.failed_closed",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "error_chain": _error_chain(exc),
            },
        )
        state.uncertainties.append(str(exc))
        return commit_terminal(
            run_id=run_id,
            store=store,
            state=state,
            status="needs_review",
        )
