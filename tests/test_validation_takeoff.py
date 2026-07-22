from __future__ import annotations

from fractions import Fraction

import pytest

from fab_agent.domain.design import FabricationDesign, ObservedComponent
from fab_agent.domain.takeoff import compute_takeoff
from fab_agent.domain.validation import validate_design
from fab_agent.infrastructure.catalogs import CatalogBundle


def test_valid_design_and_takeoff(valid_design: FabricationDesign, catalogs: CatalogBundle) -> None:
    valid_design.observed_components = [
        ObservedComponent(
            description_raw="two couplings",
            quantity=2,
            nominal_size_raw='4"',
            kind="Coupling",
        )
    ]
    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )
    assert report.passed
    assert report.geometries[0].total_length.display == "8' 0\""

    takeoff = compute_takeoff(valid_design, report)
    assert takeoff.component_summary == {'1 1/4" threaded_outlet': 1, '4" coupling': 2}
    assert takeoff.observed_component_summary == {'4" coupling': 2}
    assert takeoff.reconciliation_warnings == []
    assert takeoff.spools[0].segment_lengths[0].display == "3' 0\""


def test_observed_loose_parts_do_not_require_invented_geometry(
    valid_design: FabricationDesign, catalogs: CatalogBundle
) -> None:
    valid_design.spools[0].features[0].kind = "start"
    valid_design.spools[0].features[-1].kind = "end"
    valid_design.observed_components = [
        ObservedComponent(
            description_raw="2 - 4 inch couplings",
            quantity=2,
            nominal_size_raw='4"',
            kind="Coupling",
        )
    ]
    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    takeoff = compute_takeoff(valid_design, report)

    assert takeoff.component_summary == {'1 1/4" threaded_outlet': 1}
    assert takeoff.observed_component_summary == {'4" coupling': 2}
    assert takeoff.reconciliation_warnings == []


def test_conflicting_observed_and_modeled_quantities_are_reported(
    valid_design: FabricationDesign, catalogs: CatalogBundle
) -> None:
    valid_design.observed_components = [
        ObservedComponent(
            description_raw="3 couplings",
            quantity=3,
            nominal_size_raw='4"',
            kind="Coupling",
        )
    ]
    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    takeoff = compute_takeoff(valid_design, report)

    assert takeoff.reconciliation_warnings == [
        'Observed parts list quantity for 4" coupling is 3; geometry implies 2'
    ]


def test_demo_catalog_is_rejected_outside_demo_mode(
    valid_design: FabricationDesign, catalogs: CatalogBundle
) -> None:
    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=False,
    )
    assert not report.passed
    assert any(issue.code == "catalog.pipe_missing" for issue in report.issues)


@pytest.mark.parametrize("schedule", ["10", "SCH 10", "sch. 10", "Schedule 10"])
def test_schedule_source_variants_match_the_same_catalog_entry(
    schedule: str,
    valid_design: FabricationDesign,
    catalogs: CatalogBundle,
) -> None:
    valid_design.spools[0].schedule_raw = schedule
    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    assert report.passed


def test_quote_backed_demo_material_label_matches_catalog(
    valid_design: FabricationDesign,
    catalogs: CatalogBundle,
) -> None:
    valid_design.spools[0].material_raw = "A135 black steel"

    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    assert report.passed


def test_invalid_dimension_does_not_create_chain_or_total_conflict_cascades(
    valid_design: FabricationDesign,
    catalogs: CatalogBundle,
) -> None:
    valid_design.spools[0].segments[0].length_raw = "unreadable"
    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    codes = {issue.code for issue in report.issues}
    assert "dimension.invalid" in codes
    assert "segment.chain_disconnected" not in codes
    assert "pipe.total_conflict" not in codes


def test_total_conflict_does_not_create_out_of_bounds_cascade(
    valid_design: FabricationDesign,
    catalogs: CatalogBundle,
) -> None:
    valid_design.spools[0].stated_total_length_raw = "7 ft"

    report = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )

    codes = {issue.code for issue in report.issues}
    assert "pipe.total_conflict" in codes
    assert "feature.out_of_bounds" not in codes


def test_takeoff_refuses_invalid_design(catalogs: CatalogBundle) -> None:
    design = FabricationDesign()
    report = validate_design(
        design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )
    with pytest.raises(ValueError, match="passing validation"):
        compute_takeoff(design, report)
