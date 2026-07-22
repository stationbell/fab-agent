from __future__ import annotations

from fractions import Fraction

from fab_agent.domain.design import ObservedComponent
from fab_agent.domain.validation import validate_design
from fab_agent.infrastructure.catalogs import CatalogBundle
from fab_agent.models.pipe_spool import convert_observation
from fab_agent.models.schemas import (
    ObservedPhysicalFeature,
    ObservedPipeSpool,
    PipeSpoolObservation,
)


def _sample_observation(material: str | None = "demo carbon steel") -> PipeSpoolObservation:
    return PipeSpoolObservation(
        project_reference_raw="CubeSmart - 12300 College Pkwy",
        spools=[
            ObservedPipeSpool(
                id_raw="S1",
                nominal_size_raw='4"',
                schedule_raw="10",
                material_raw=material,
                stated_total_length_raw="8'-0\"",
                physical_features_left_to_right=[
                    ObservedPhysicalFeature(kind="start", label="G"),
                    ObservedPhysicalFeature(
                        kind="outlet",
                        nominal_size_raw='1 1/2"',
                        connection_type="threaded",
                        orientation_raw="A",
                        orientation="up",
                        label="A",
                    ),
                    ObservedPhysicalFeature(
                        kind="outlet",
                        nominal_size_raw='1 1/4"',
                        connection_type="threaded",
                        orientation_raw="B",
                        orientation="right",
                        label="B",
                    ),
                    ObservedPhysicalFeature(kind="end", label="P"),
                ],
                segment_lengths_left_to_right=[
                    "1'-9 1/4\"",
                    "4'-6 1/4\"",
                    "1'-8 1/2\"",
                ],
            )
        ],
    )


def test_python_builds_feature_ids_and_segment_topology() -> None:
    inspection = convert_observation(_sample_observation())
    spool = inspection.design.spools[0]

    assert [feature.id for feature in spool.features] == ["f1", "f2", "f3", "f4"]
    assert [(segment.from_feature_id, segment.to_feature_id) for segment in spool.segments] == [
        ("f1", "f2"),
        ("f2", "f3"),
        ("f3", "f4"),
    ]
    assert [segment.length_raw for segment in spool.segments] == [
        "1'-9 1/4\"",
        "4'-6 1/4\"",
        "1'-8 1/2\"",
    ]


def test_python_built_topology_passes_deterministic_validation(
    catalogs: CatalogBundle,
) -> None:
    design = convert_observation(_sample_observation()).design
    report = validate_design(
        design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    assert report.passed is True
    assert report.geometries[0].positions["f2"].display == "1' 9 1/4\""
    assert report.geometries[0].positions["f3"].display == "6' 3 1/2\""


def test_generic_pipe_word_is_not_accepted_as_material() -> None:
    inspection = convert_observation(_sample_observation(material="pipe"))

    assert inspection.design.spools[0].material_raw is None
    assert inspection.uncertainties == [
        "Spool S1: Ignored non-material placeholder 'pipe'; pipe material remains missing."
    ]


def test_segment_count_mismatch_is_visible_and_not_invented() -> None:
    observation = _sample_observation()
    observation.spools[0].segment_lengths_left_to_right.pop()
    inspection = convert_observation(observation)

    assert len(inspection.design.spools[0].segments) == 2
    assert "expected 3" in inspection.uncertainties[0]


def test_unknown_boundary_kinds_become_topological_start_and_end() -> None:
    observation = _sample_observation()
    observation.spools[0].physical_features_left_to_right[0].kind = "unknown_end"
    observation.spools[0].physical_features_left_to_right[-1].kind = "unknown_end"

    inspection = convert_observation(observation)

    assert inspection.design.spools[0].features[0].kind == "start"
    assert inspection.design.spools[0].features[-1].kind == "end"
    assert sum("normalized boundary feature" in item for item in inspection.uncertainties) == 2


def test_explicit_parts_list_word_recovers_missing_component_kind() -> None:
    observation = _sample_observation()
    observation.observed_components = [
        ObservedComponent(
            description_raw='4" Couplings',
            quantity=2,
            nominal_size_raw='4"',
        )
    ]

    inspection = convert_observation(observation)

    assert inspection.design.observed_components[0].kind == "coupling"
    assert "explicit parts-list text" in inspection.uncertainties[0]
