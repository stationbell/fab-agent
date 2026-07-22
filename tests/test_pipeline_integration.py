from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from conftest import FakeModel

from fab_agent.api import resume_fab_agent, run_fab_agent
from fab_agent.application.pipeline import _error_chain, vision_prompt
from fab_agent.application.runner import Dependencies, rebuild_run
from fab_agent.config import AppConfig
from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.results import FabRequest
from fab_agent.errors import ModelError, RunStateError
from fab_agent.infrastructure.catalogs import CatalogBundle
from fab_agent.infrastructure.filesystem import FilesystemRunStore
from fab_agent.infrastructure.runtime import SystemClock
from fab_agent.models.schemas import (
    ObservedPhysicalFeature,
    ObservedPipeSpool,
    PipeSpoolObservation,
)


def dependencies(config: AppConfig, catalogs: CatalogBundle, model: FakeModel) -> Dependencies:
    return Dependencies(
        config=config,
        model=model,  # type: ignore[arg-type]
        store=FilesystemRunStore(config.output.root),
        catalogs=catalogs,
        clock=SystemClock(),
    )


def observation(
    design: FabricationDesign,
    *,
    uncertainties: list[str] | None = None,
) -> PipeSpoolObservation:
    return PipeSpoolObservation(
        project_reference_raw=design.project_reference_raw,
        spools=[
            ObservedPipeSpool(
                id_raw=spool.id,
                nominal_size_raw=spool.nominal_size_raw,
                schedule_raw=spool.schedule_raw,
                material_raw=spool.material_raw,
                stated_total_length_raw=spool.stated_total_length_raw,
                physical_features_left_to_right=[
                    ObservedPhysicalFeature(**feature.model_dump(exclude={"id"}, exclude_none=True))
                    for feature in spool.features
                ],
                segment_lengths_left_to_right=[segment.length_raw for segment in spool.segments],
                orientation_legend=spool.orientation_legend,
                notes=spool.notes,
            )
            for spool in design.spools
        ],
        observed_components=design.observed_components,
        notes=design.notes,
        uncertainties=uncertainties or [],
    )


def test_vision_prompt_defines_the_transcription_contract() -> None:
    prompt = vision_prompt()
    assert "Python owns all feature IDs and segment links" in prompt
    assert "must not become" in prompt
    assert "connection_type" in prompt
    assert "do not calculate dimensions" in prompt
    assert "physical outlet stem visibly points" in prompt
    assert "exclude field labels such as Total" in prompt


def test_error_chain_records_types_without_persisting_exception_messages() -> None:
    path = Path.cwd() / "private-input.png"
    try:
        try:
            raise FileNotFoundError(path)
        except FileNotFoundError as exc:
            raise ValueError("safe outer message") from exc
    except ValueError as exc:
        chain = _error_chain(exc)

    assert chain == ["ValueError", "FileNotFoundError"]
    assert str(path) not in repr(chain)


def test_unknown_design_type_fails_before_creating_a_run(
    app_config: AppConfig, catalogs: CatalogBundle, input_image: Path
) -> None:
    model = FakeModel([])
    with pytest.raises(RunStateError, match="Unsupported design type"):
        asyncio.run(
            run_fab_agent(
                FabRequest(input_image=input_image, design_type="unknown"),
                dependencies(app_config, catalogs, model),
            )
        )
    assert model.calls == 0
    assert not app_config.output.root.exists()


def test_clear_design_uses_one_model_call_and_completes(
    app_config: AppConfig,
    catalogs: CatalogBundle,
    valid_design: FabricationDesign,
    input_image: Path,
) -> None:
    model = FakeModel([observation(valid_design)])
    result = asyncio.run(
        run_fab_agent(
            FabRequest(input_image=input_image, demo=True),
            dependencies(app_config, catalogs, model),
        )
    )

    assert result.status == "complete"
    assert model.calls == 1
    assert model.responses == []
    assert result.artifacts["bom"].is_file()
    assert result.artifacts["spool-001_png"].is_file()
    assert "NOT APPROVED FOR FABRICATION" in result.artifacts["review"].read_text()
    assert (result.run_path / "versions" / "001" / "manifest.toml").is_file()
    events = (result.run_path / "events.jsonl").read_text()
    assert events.count("pipeline.extraction_completed") == 1
    assert "pipeline.validation_completed" in events
    assert "pipeline.outputs_generated" in events


def test_missing_material_pauses_deterministically_and_resume_uses_no_model(
    app_config: AppConfig,
    catalogs: CatalogBundle,
    valid_design: FabricationDesign,
    input_image: Path,
) -> None:
    valid_design.spools[0].material_raw = None
    first_model = FakeModel([observation(valid_design, uncertainties=["material missing"])])
    deps = dependencies(app_config, catalogs, first_model)
    paused = asyncio.run(run_fab_agent(FabRequest(input_image=input_image, demo=True), deps))

    assert paused.status == "awaiting_input"
    assert paused.question == "What is the pipe material for spool-001?"
    assert first_model.calls == 1
    assert "demo_mode = true" in (paused.run_path / "run.toml").read_text()
    version_one = paused.run_path / "versions" / "001" / "design.toml"
    original_snapshot = version_one.read_bytes()

    no_model = FakeModel([])
    resumed = asyncio.run(
        resume_fab_agent(
            paused.run_id,
            "demo carbon steel",
            dependencies(app_config, catalogs, no_model),
        )
    )

    assert resumed.status == "complete"
    assert no_model.calls == 0
    assert (paused.run_path / "versions" / "002" / "design.toml").is_file()
    assert version_one.read_bytes() == original_snapshot
    assert "demo carbon steel" not in version_one.read_text()
    assert "demo carbon steel" in (paused.run_path / "versions" / "002" / "design.toml").read_text()


def test_invalid_geometry_fails_review_without_an_orchestrator(
    app_config: AppConfig,
    catalogs: CatalogBundle,
    valid_design: FabricationDesign,
    input_image: Path,
) -> None:
    valid_design.spools[0].stated_total_length_raw = "7 ft"
    model = FakeModel([observation(valid_design)])
    result = asyncio.run(
        run_fab_agent(
            FabRequest(input_image=input_image, demo=True),
            dependencies(app_config, catalogs, model),
        )
    )

    assert result.status == "needs_review"
    assert model.calls == 1
    assert any("does not match segment chain" in warning for warning in result.warnings)


def test_malformed_model_output_fails_closed(
    app_config: AppConfig, catalogs: CatalogBundle, input_image: Path
) -> None:
    model = FakeModel([ModelError("malformed structured output")])
    result = asyncio.run(
        run_fab_agent(
            FabRequest(input_image=input_image, demo=True),
            dependencies(app_config, catalogs, model),
        )
    )

    assert result.status == "needs_review"
    assert model.calls == 1
    assert "malformed structured output" in result.warnings
    events = (result.run_path / "events.jsonl").read_text()
    assert "pipeline.failed_closed" in events
    # The event stream names the exception type so that a defect is
    # distinguishable from an unreadable sketch after the fact.
    assert '"error_type":"ModelError"' in events
    assert str(Path.cwd()) not in events


def test_extraction_has_a_hard_wall_clock_timeout(
    app_config: AppConfig, catalogs: CatalogBundle, input_image: Path
) -> None:
    class SlowModel(FakeModel):
        async def extract_image(self, **kwargs: object) -> PipeSpoolObservation:
            del kwargs
            self.calls += 1
            await asyncio.sleep(2)
            raise AssertionError("hard timeout did not cancel the model call")

    config = app_config.model_copy(
        update={"ollama": app_config.ollama.model_copy(update={"request_timeout_seconds": 1})}
    )
    model = SlowModel([])
    result = asyncio.run(
        run_fab_agent(
            FabRequest(input_image=input_image, demo=True),
            dependencies(config, catalogs, model),
        )
    )

    assert result.status == "needs_review"
    assert model.calls == 1
    assert any("after 1 seconds" in warning for warning in result.warnings)


def test_rebuild_is_deterministic_and_does_not_call_model(
    app_config: AppConfig,
    catalogs: CatalogBundle,
    valid_design: FabricationDesign,
    input_image: Path,
) -> None:
    initial_model = FakeModel([observation(valid_design)])
    deps = dependencies(app_config, catalogs, initial_model)
    first = asyncio.run(run_fab_agent(FabRequest(input_image=input_image, demo=True), deps))
    design, provenance = deps.store.load_draft(first.run_id)
    design.project_reference_raw = "Manually corrected project"
    deps.store.save_draft(first.run_id, design, provenance)

    no_model = FakeModel([])
    rebuilt = asyncio.run(rebuild_run(first.run_id, dependencies(app_config, catalogs, no_model)))

    assert rebuilt.status == "complete"
    assert no_model.calls == 0
    assert (first.run_path / "versions" / "002").is_dir()
    provenance_text = (first.run_path / "versions" / "002" / "provenance.toml").read_text()
    assert "manual_edit" in provenance_text
