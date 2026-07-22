"""Strict schemas for every model-authored response."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from fab_agent.domain.design import (
    DomainModel,
    ObservedComponent,
    OrientationLegendEntry,
)


class ObservedPhysicalFeature(DomainModel):
    """One physical feature in left-to-right order; identifiers are Python-owned."""

    kind: Literal[
        "start",
        "end",
        "outlet",
        "coupling",
        "cap",
        "open_end",
        "unknown_end",
    ]
    position_raw: str | None = None
    nominal_size_raw: str | None = None
    connection_type: Literal["threaded", "grooved"] | None = None
    orientation_raw: str | None = None
    orientation: Literal["up", "right", "down", "left"] | None = None
    label: str | None = None


class ObservedPipeSpool(DomainModel):
    """Flat visual observation; ordered dimensions sit between ordered features."""

    id_raw: str | None = None
    nominal_size_raw: str | None = None
    schedule_raw: str | None = None
    material_raw: str | None = None
    stated_total_length_raw: str | None = None
    physical_features_left_to_right: list[ObservedPhysicalFeature] = Field(default_factory=list)
    segment_lengths_left_to_right: list[str] = Field(default_factory=list)
    orientation_legend: list[OrientationLegendEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PipeSpoolObservation(DomainModel):
    """The complete model-authored response for the pipe-spool handler."""

    project_reference_raw: str | None = None
    spools: list[ObservedPipeSpool] = Field(default_factory=list)
    observed_components: list[ObservedComponent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
