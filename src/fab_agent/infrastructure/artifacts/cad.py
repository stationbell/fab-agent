"""Simplified CadQuery STEP review models."""

from __future__ import annotations

from pathlib import Path

from fab_agent.domain.design import Spool
from fab_agent.domain.dimensions import parse_dimension
from fab_agent.domain.validation import SpoolGeometry
from fab_agent.errors import ArtifactError
from fab_agent.infrastructure.catalogs import CatalogBundle


def _millimeters(inches: float) -> float:
    return inches * 25.4


def generate_step(
    path: Path,
    spool: Spool,
    geometry: SpoolGeometry,
    catalogs: CatalogBundle,
    *,
    allow_demo: bool,
) -> None:
    """Generate deliberately simplified cylinders for visual review only."""

    try:
        import cadquery as cq
        from cadquery.occ_impl.exporters.assembly import exportAssembly

        pipe_entry = next(
            entry for entry in catalogs.pipes if entry.key == geometry.pipe_catalog_key
        )
        outer_radius = _millimeters(
            float(parse_dimension(pipe_entry.outside_diameter_in).inches / 2)
        )
        wall = _millimeters(float(parse_dimension(pipe_entry.wall_thickness_in).inches))
        inner_radius = max(outer_radius - wall, 0.1)
        length = _millimeters(float(geometry.total_length.as_fraction()))
        body = cq.Workplane("YZ").circle(outer_radius).circle(inner_radius).extrude(length)

        for feature in spool.features:
            if feature.kind in {"coupling", "cap"} and feature.nominal_size_raw:
                entry = catalogs.find_component(
                    feature.kind, feature.nominal_size_raw, allow_demo=allow_demo
                )
                if entry is None:
                    raise ArtifactError(f"Missing CAD catalog entry for {feature.id}")
                component_radius = _millimeters(
                    float(parse_dimension(entry.outside_diameter_in).inches / 2)
                )
                component_length = _millimeters(float(parse_dimension(entry.length_in).inches))
                position = _millimeters(float(geometry.positions[feature.id].as_fraction()))
                origin = position - component_length / 2
                component = (
                    cq.Workplane("YZ", origin=(origin, 0, 0))
                    .circle(component_radius)
                    .extrude(component_length)
                )
                body = body.union(component)
                continue
            if (
                feature.kind != "outlet"
                or not feature.nominal_size_raw
                or not feature.connection_type
            ):
                continue
            entry = catalogs.find_component(
                f"{feature.connection_type}_outlet",
                feature.nominal_size_raw,
                allow_demo=allow_demo,
            )
            if entry is None:
                raise ArtifactError(f"Missing CAD catalog entry for outlet {feature.id}")
            position = _millimeters(float(geometry.positions[feature.id].as_fraction()))
            radius = _millimeters(float(parse_dimension(entry.outside_diameter_in).inches / 2))
            outlet_length = _millimeters(float(parse_dimension(entry.length_in).inches))
            if feature.orientation in {"up", "down"}:
                outlet = (
                    cq.Workplane("XY", origin=(position, 0, 0))
                    .circle(radius)
                    .extrude(outlet_length if feature.orientation == "up" else -outlet_length)
                )
            else:
                sign = 1 if feature.orientation == "right" else -1
                outlet = (
                    cq.Workplane("XZ", origin=(position, 0, 0))
                    .circle(radius)
                    .extrude(sign * outlet_length)
                )
            body = body.union(outlet)
        path.parent.mkdir(parents=True, exist_ok=True)
        assembly = cq.Assembly(name="REVIEW_OUTPUT_NOT_APPROVED_FOR_FABRICATION")
        assembly.add(body, name=f"REVIEW_ONLY_{spool.id}")
        if not exportAssembly(assembly, str(path)):
            raise ArtifactError(f"CadQuery did not export STEP for {spool.id}")
    except ArtifactError:
        raise
    except (ImportError, OSError, StopIteration, ValueError, RuntimeError) as exc:
        raise ArtifactError(f"CadQuery STEP generation failed for {spool.id}: {exc}") from exc
