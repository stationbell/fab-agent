"""Human-editable source design schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(DomainModel):
    id: str
    from_feature_id: str
    to_feature_id: str
    length_raw: str


class Feature(DomainModel):
    id: str
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


class OrientationLegendEntry(DomainModel):
    label: str
    direction: Literal["up", "right", "down", "left"]
    raw_text: str | None = None


class Spool(DomainModel):
    id: str
    nominal_size_raw: str | None = None
    schedule_raw: str | None = None
    material_raw: str | None = None
    stated_total_length_raw: str | None = None
    features: list[Feature] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    orientation_legend: list[OrientationLegendEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ObservedComponent(DomainModel):
    description_raw: str
    quantity: PositiveInt
    nominal_size_raw: str | None = None
    kind: str | None = None
    code_raw: str | None = None


class FabricationDesign(DomainModel):
    schema_version: Literal[1] = 1
    project_reference_raw: str | None = None
    spools: list[Spool] = Field(default_factory=list)
    observed_components: list[ObservedComponent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
