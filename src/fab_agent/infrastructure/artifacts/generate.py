"""Package-level deterministic artifact generation."""

from __future__ import annotations

import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.takeoff import Takeoff
from fab_agent.domain.validation import ValidationReport
from fab_agent.errors import ArtifactError
from fab_agent.infrastructure.catalogs import CatalogBundle

REVIEW_LABEL = "REVIEW OUTPUT — NOT APPROVED FOR FABRICATION"


@dataclass(frozen=True, slots=True)
class ArtifactSet:
    root: Path
    files: dict[str, Path]


def _write_bom(path: Path, takeoff: Takeoff) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_status", "source", "item", "quantity"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "review_status": REVIEW_LABEL,
                "source": "derived_geometry",
                "item": "main_pipe",
                "quantity": len(takeoff.spools),
            }
        )
        modeled = takeoff.component_summary
        observed = takeoff.observed_component_summary
        for item in sorted(modeled.keys() | observed.keys()):
            modeled_quantity = modeled.get(item)
            observed_quantity = observed.get(item)
            if modeled_quantity is not None and modeled_quantity == observed_quantity:
                writer.writerow(
                    {
                        "review_status": REVIEW_LABEL,
                        "source": "observed_and_derived",
                        "item": item,
                        "quantity": modeled_quantity,
                    }
                )
                continue
            if modeled_quantity is not None:
                writer.writerow(
                    {
                        "review_status": REVIEW_LABEL,
                        "source": "derived_geometry",
                        "item": item,
                        "quantity": modeled_quantity,
                    }
                )
            if observed_quantity is not None:
                writer.writerow(
                    {
                        "review_status": REVIEW_LABEL,
                        "source": "observed_parts_list",
                        "item": item,
                        "quantity": observed_quantity,
                    }
                )


def _write_report(
    path: Path,
    design: FabricationDesign,
    validation: ValidationReport,
    takeoff: Takeoff,
    *,
    demo: bool,
    cad_enabled: bool,
) -> None:
    lines = [
        f"# {REVIEW_LABEL}",
        "",
        f"Project: {design.project_reference_raw or 'Not provided'}",
        "",
        f"Catalog mode: {'DEMO — SYNTHETIC GEOMETRY' if demo else 'reviewed catalog'}",
        "",
        "## Spools",
        "",
    ]
    takeoff_by_id = {item.spool_id: item for item in takeoff.spools}
    for spool in design.spools:
        item = takeoff_by_id[spool.id]
        lines.extend(
            [
                f"### {spool.id}",
                "",
                f"- Pipe: {spool.nominal_size_raw}, schedule {spool.schedule_raw}",
                f"- Material: {spool.material_raw}",
                f"- Total length: {item.main_run_length.display}",
                f"- Features: {sum(item.component_counts.values())}",
                "",
            ]
        )
    if cad_enabled:
        lines.extend(
            [
                "## CAD compatibility",
                "",
                "- Open each `spools/<spool-id>.step` file in Autodesk Fusion as solid-body "
                "review geometry.",
                "- STEP does not contain a native Fusion parametric timeline.",
                "",
            ]
        )
    lines.extend(["## Validation", ""])
    if validation.issues:
        lines.extend(f"- {issue.level.upper()}: {issue.message}" for issue in validation.issues)
    else:
        lines.append("- Deterministic V1 validation passed.")
    lines.extend(["", "## Parts-list reconciliation", ""])
    if takeoff.reconciliation_warnings:
        lines.extend(f"- WARNING: {warning}" for warning in takeoff.reconciliation_warnings)
    else:
        lines.append(
            "- No quantity conflicts were found where observed and modeled component kinds overlap."
        )
    if takeoff.observed_component_summary:
        lines.append(
            "- Observed parts without modeled placement are retained in the BOM as "
            "`observed_parts_list`; they are not placed in CAD."
        )
    lines.extend(["", f"**{REVIEW_LABEL}**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_artifacts(
    design: FabricationDesign,
    validation: ValidationReport,
    takeoff: Takeoff,
    catalogs: CatalogBundle,
    *,
    allow_demo: bool,
    cad_enabled: bool,
) -> ArtifactSet:
    if not validation.passed:
        raise ArtifactError("Fabrication artifacts require passing validation")
    root = Path(tempfile.mkdtemp(prefix="fab-agent-artifacts-"))
    files: dict[str, Path] = {}
    try:
        from fab_agent.infrastructure.artifacts.diagrams import generate_diagram

        geometry_by_id = {geometry.spool_id: geometry for geometry in validation.geometries}
        for spool in design.spools:
            geometry = geometry_by_id[spool.id]
            spool_root = root / "spools"
            png = spool_root / f"{spool.id}.png"
            svg = spool_root / f"{spool.id}.svg"
            generate_diagram(png, spool, geometry, validation.issues)
            generate_diagram(svg, spool, geometry, validation.issues)
            files[f"{spool.id}_png"] = png
            files[f"{spool.id}_svg"] = svg
            if cad_enabled:
                from fab_agent.infrastructure.artifacts.cad import generate_step

                step = spool_root / f"{spool.id}.step"
                generate_step(step, spool, geometry, catalogs, allow_demo=allow_demo)
                files[f"{spool.id}_step"] = step
        bom = root / "bom.csv"
        report = root / "review.md"
        _write_bom(bom, takeoff)
        _write_report(
            report,
            design,
            validation,
            takeoff,
            demo=allow_demo,
            cad_enabled=cad_enabled,
        )
        files["bom"] = bom
        files["review"] = report
        return ArtifactSet(root=root, files=files)
    except ArtifactError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except (OSError, ValueError) as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise ArtifactError(f"Artifact generation failed: {exc}") from exc
