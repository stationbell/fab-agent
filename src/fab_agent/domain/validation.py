"""Deterministic structural and catalog validation."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

from pydantic import Field

from fab_agent.domain.design import DomainModel, FabricationDesign, Feature, Spool
from fab_agent.domain.dimensions import format_inches, parse_dimension
from fab_agent.errors import DimensionParseError
from fab_agent.infrastructure.catalogs import CatalogBundle


class RationalValue(DomainModel):
    numerator: int
    denominator: int = Field(gt=0)
    display: str

    @classmethod
    def from_fraction(cls, value: Fraction) -> RationalValue:
        return cls(
            numerator=value.numerator,
            denominator=value.denominator,
            display=format_inches(value),
        )

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class ValidationIssue(DomainModel):
    level: Literal["error", "warning", "info"]
    code: str
    message: str
    spool_id: str | None = None
    field_path: str | None = None


class SpoolGeometry(DomainModel):
    spool_id: str
    total_length: RationalValue
    positions: dict[str, RationalValue] = Field(default_factory=dict)
    pipe_catalog_key: str


class ValidationReport(DomainModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    geometries: list[SpoolGeometry] = Field(default_factory=list)


def _issue(
    issues: list[ValidationIssue],
    level: Literal["error", "warning", "info"],
    code: str,
    message: str,
    spool: Spool | None = None,
    field_path: str | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            level=level,
            code=code,
            message=message,
            spool_id=spool.id if spool else None,
            field_path=field_path,
        )
    )


def _parse_source_dimension(
    value: str | None,
    *,
    issues: list[ValidationIssue],
    spool: Spool,
    field_path: str,
) -> Fraction | None:
    if not value:
        return None
    try:
        return parse_dimension(value).inches
    except DimensionParseError as exc:
        _issue(issues, "error", "dimension.invalid", str(exc), spool, field_path)
        return None


def _resolve_chain(
    spool: Spool, issues: list[ValidationIssue]
) -> tuple[Fraction | None, dict[str, Fraction]]:
    if not spool.segments:
        return None, {}
    positions: dict[str, Fraction] = {}
    running = Fraction(0)
    previous_to: str | None = None
    chain_complete = True
    for index, segment in enumerate(spool.segments):
        path = f"spools.{spool.id}.segments.{segment.id}.length_raw"
        if index == 0:
            positions[segment.from_feature_id] = Fraction(0)
        elif segment.from_feature_id != previous_to:
            _issue(
                issues,
                "error",
                "segment.chain_disconnected",
                f"Segment {segment.id} does not start at {previous_to}",
                spool,
                path,
            )
            chain_complete = False
        previous_to = segment.to_feature_id
        length = _parse_source_dimension(
            segment.length_raw, issues=issues, spool=spool, field_path=path
        )
        if length is None:
            chain_complete = False
            continue
        if length <= 0:
            _issue(
                issues,
                "error",
                "segment.non_positive",
                "Segment length must be positive",
                spool,
                path,
            )
            chain_complete = False
        if not chain_complete:
            continue
        if segment.from_feature_id not in positions:
            positions[segment.from_feature_id] = running
        running += length
        positions[segment.to_feature_id] = running
    return (running if chain_complete else None), positions


def _validate_features(
    spool: Spool,
    total: Fraction,
    chain_positions: dict[str, Fraction],
    tolerance: Fraction,
    catalogs: CatalogBundle,
    allow_demo: bool,
    issues: list[ValidationIssue],
) -> dict[str, RationalValue]:
    positions = dict(chain_positions)
    feature_ids = [feature.id for feature in spool.features]
    if len(feature_ids) != len(set(feature_ids)):
        _issue(issues, "error", "feature.duplicate_id", "Feature IDs must be unique", spool)
    known_ids = set(feature_ids)
    for segment in spool.segments:
        for feature_id in (segment.from_feature_id, segment.to_feature_id):
            if feature_id not in known_ids:
                _issue(
                    issues,
                    "error",
                    "segment.unknown_feature",
                    f"Segment {segment.id} references unknown feature {feature_id}",
                    spool,
                )

    outlet_positions: list[tuple[str, Fraction]] = []
    for feature in spool.features:
        path = f"spools.{spool.id}.features.{feature.id}"
        explicit = _parse_source_dimension(
            feature.position_raw,
            issues=issues,
            spool=spool,
            field_path=f"{path}.position_raw",
        )
        derived = positions.get(feature.id)
        if explicit is not None and derived is not None and abs(explicit - derived) > tolerance:
            _issue(
                issues,
                "error",
                "feature.position_conflict",
                f"Explicit and chain positions disagree for {feature.id}",
                spool,
                path,
            )
        position = explicit if explicit is not None else derived
        if position is None:
            _issue(
                issues,
                "error",
                "feature.position_missing",
                f"Position missing for {feature.id}",
                spool,
                path,
            )
        else:
            positions[feature.id] = position
            if position < 0 or position > total:
                _issue(
                    issues,
                    "error",
                    "feature.out_of_bounds",
                    f"{feature.id} falls outside the main run",
                    spool,
                    path,
                )

        if feature.kind == "outlet":
            _validate_outlet(feature, spool, path, catalogs, allow_demo, issues)
            if position is not None:
                outlet_positions.append((feature.id, position))
        if feature.kind in {"coupling", "cap"}:
            nominal_size = feature.nominal_size_raw
            if not nominal_size:
                _issue(
                    issues,
                    "error",
                    "component.size_missing",
                    f"Component size missing for {feature.id}",
                    spool,
                    path,
                )
            else:
                try:
                    component_entry = catalogs.find_component(
                        feature.kind, nominal_size, allow_demo=allow_demo
                    )
                except DimensionParseError as exc:
                    _issue(issues, "error", "component.size_invalid", str(exc), spool, path)
                else:
                    if component_entry is None:
                        _issue(
                            issues,
                            "error",
                            "catalog.component_missing",
                            f"No catalog geometry for {feature.kind} {nominal_size}",
                            spool,
                            path,
                        )
        if feature.kind == "unknown_end":
            _issue(
                issues,
                "error",
                "end.unresolved",
                f"End component {feature.id} is unresolved",
                spool,
                path,
            )

    for index, (left_id, left_position) in enumerate(outlet_positions):
        for right_id, right_position in outlet_positions[index + 1 :]:
            if abs(left_position - right_position) <= tolerance:
                _issue(
                    issues,
                    "error",
                    "outlet.position_conflict",
                    f"Outlets {left_id} and {right_id} conflict within tolerance",
                    spool,
                )
    return {key: RationalValue.from_fraction(value) for key, value in positions.items()}


def _validate_outlet(
    feature: Feature,
    spool: Spool,
    path: str,
    catalogs: CatalogBundle,
    allow_demo: bool,
    issues: list[ValidationIssue],
) -> None:
    if not feature.nominal_size_raw:
        _issue(
            issues,
            "error",
            "outlet.size_missing",
            f"Outlet size missing for {feature.id}",
            spool,
            path,
        )
    if not feature.connection_type:
        _issue(
            issues,
            "error",
            "outlet.connection_missing",
            f"Connection type missing for {feature.id}",
            spool,
            path,
        )
    if not feature.orientation:
        _issue(
            issues,
            "error",
            "outlet.orientation_missing",
            f"Orientation missing for {feature.id}",
            spool,
            path,
        )
    if feature.nominal_size_raw and feature.connection_type:
        catalog_kind = f"{feature.connection_type}_outlet"
        try:
            entry = catalogs.find_component(
                catalog_kind, feature.nominal_size_raw, allow_demo=allow_demo
            )
        except DimensionParseError as exc:
            _issue(issues, "error", "outlet.size_invalid", str(exc), spool, path)
        else:
            if entry is None:
                _issue(
                    issues,
                    "error",
                    "catalog.component_missing",
                    f"No catalog geometry for {catalog_kind} {feature.nominal_size_raw}",
                    spool,
                    path,
                )


def validate_design(
    design: FabricationDesign,
    *,
    tolerance: Fraction,
    catalogs: CatalogBundle,
    allow_demo: bool = False,
) -> ValidationReport:
    """Validate source design and resolve only defensible exact geometry."""

    issues: list[ValidationIssue] = []
    geometries: list[SpoolGeometry] = []
    if not design.spools:
        _issue(issues, "error", "design.no_spools", "At least one straight spool is required")
    spool_ids = [spool.id for spool in design.spools]
    if len(spool_ids) != len(set(spool_ids)):
        _issue(issues, "error", "spool.duplicate_id", "Spool IDs must be unique")

    for spool in design.spools:
        if not spool.nominal_size_raw:
            _issue(issues, "error", "pipe.size_missing", "Nominal pipe size is required", spool)
        if not spool.schedule_raw:
            _issue(issues, "error", "pipe.schedule_missing", "Pipe schedule is required", spool)
        if not spool.material_raw:
            _issue(issues, "error", "pipe.material_missing", "Pipe material is required", spool)

        chain_total, chain_positions = _resolve_chain(spool, issues)
        stated_total = _parse_source_dimension(
            spool.stated_total_length_raw,
            issues=issues,
            spool=spool,
            field_path=f"spools.{spool.id}.stated_total_length_raw",
        )
        if stated_total is not None and stated_total <= 0:
            _issue(
                issues, "error", "pipe.total_non_positive", "Total length must be positive", spool
            )
        if (
            stated_total is not None
            and chain_total is not None
            and abs(stated_total - chain_total) > tolerance
        ):
            _issue(
                issues,
                "error",
                "pipe.total_conflict",
                f"Stated total {format_inches(stated_total)} does not match segment chain "
                f"{format_inches(chain_total)}",
                spool,
            )
        total = stated_total if stated_total is not None else chain_total
        if total is None:
            _issue(
                issues,
                "error",
                "pipe.total_missing",
                "A stated total or complete segment chain is required",
                spool,
            )
            continue

        pipe_entry = None
        if spool.nominal_size_raw and spool.schedule_raw and spool.material_raw:
            try:
                pipe_entry = catalogs.find_pipe(
                    spool.nominal_size_raw,
                    spool.schedule_raw,
                    spool.material_raw,
                    allow_demo=allow_demo,
                )
            except DimensionParseError as exc:
                _issue(issues, "error", "pipe.size_invalid", str(exc), spool)
            if pipe_entry is None:
                _issue(
                    issues,
                    "error",
                    "catalog.pipe_missing",
                    "No matching pipe catalog geometry for size, schedule, and material",
                    spool,
                )
        positions = _validate_features(
            spool,
            total,
            chain_positions,
            tolerance,
            catalogs,
            allow_demo,
            issues,
        )
        spool_has_error = any(
            issue.level == "error" and issue.spool_id == spool.id for issue in issues
        )
        if pipe_entry is not None and not spool_has_error:
            geometries.append(
                SpoolGeometry(
                    spool_id=spool.id,
                    total_length=RationalValue.from_fraction(total),
                    positions=positions,
                    pipe_catalog_key=pipe_entry.key,
                )
            )

    return ValidationReport(
        passed=not any(issue.level == "error" for issue in issues),
        issues=issues,
        geometries=geometries,
    )
