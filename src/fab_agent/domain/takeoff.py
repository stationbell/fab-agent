"""Reproducible takeoff derived only from validated source data."""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from fab_agent.domain.design import DomainModel, FabricationDesign
from fab_agent.domain.dimensions import parse_dimension
from fab_agent.domain.validation import RationalValue, ValidationReport


def _normalize_component_kind(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split())


class FeaturePosition(DomainModel):
    feature_id: str
    position: RationalValue


class SpoolTakeoff(DomainModel):
    spool_id: str
    main_pipe_quantity: int = 1
    main_run_length: RationalValue
    feature_positions: list[FeaturePosition] = Field(default_factory=list)
    segment_lengths: list[RationalValue] = Field(default_factory=list)
    component_counts: dict[str, int] = Field(default_factory=dict)


class Takeoff(DomainModel):
    schema_version: int = 1
    spools: list[SpoolTakeoff] = Field(default_factory=list)
    component_summary: dict[str, int] = Field(default_factory=dict)
    observed_component_summary: dict[str, int] = Field(default_factory=dict)
    reconciliation_warnings: list[str] = Field(default_factory=list)


def compute_takeoff(design: FabricationDesign, validation: ValidationReport) -> Takeoff:
    if not validation.passed:
        raise ValueError("Takeoff requires passing validation")
    geometry_by_id = {geometry.spool_id: geometry for geometry in validation.geometries}
    summary: Counter[str] = Counter()
    spool_takeoffs: list[SpoolTakeoff] = []
    for spool in design.spools:
        geometry = geometry_by_id[spool.id]
        counts: Counter[str] = Counter()
        for feature in spool.features:
            key: str = feature.kind
            if feature.kind == "outlet" and feature.connection_type:
                key = f"{feature.connection_type}_outlet"
            if feature.kind not in {"start", "end"}:
                counts[key] += 1
                summary[key] += 1
        segment_lengths = [
            RationalValue.from_fraction(parse_dimension(segment.length_raw).inches)
            for segment in spool.segments
        ]
        spool_takeoffs.append(
            SpoolTakeoff(
                spool_id=spool.id,
                main_run_length=geometry.total_length,
                feature_positions=[
                    FeaturePosition(feature_id=feature_id, position=position)
                    for feature_id, position in geometry.positions.items()
                ],
                segment_lengths=segment_lengths,
                component_counts=dict(sorted(counts.items())),
            )
        )
    observed: Counter[str] = Counter()
    for component in design.observed_components:
        if component.kind:
            observed[_normalize_component_kind(component.kind)] += component.quantity
    # An observed parts list may include loose components whose exact placement is
    # not drawn. Compare quantities only when the same component kind is also
    # represented in the modeled geometry. Otherwise retain the observed item as
    # a separate BOM source without inventing a CAD position for it.
    warnings = [
        f"Observed parts list has {observed[kind]} {kind}; geometry implies {summary[kind]}"
        for kind in sorted(observed.keys() & summary.keys())
        if observed[kind] != summary[kind]
    ]
    return Takeoff(
        spools=spool_takeoffs,
        component_summary=dict(sorted(summary.items())),
        observed_component_summary=dict(sorted(observed.items())),
        reconciliation_warnings=warnings,
    )
