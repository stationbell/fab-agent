"""Deterministic PNG and SVG review diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from fab_agent.domain.design import Feature, Spool
from fab_agent.domain.validation import SpoolGeometry, ValidationIssue

REVIEW_LABEL = "REVIEW OUTPUT — NOT APPROVED FOR FABRICATION"


def _feature_label(feature: Feature) -> str:
    if feature.label:
        return feature.label
    size = f"{feature.nominal_size_raw} " if feature.nominal_size_raw else ""
    if feature.kind == "outlet":
        connection = f"{feature.connection_type} " if feature.connection_type else ""
        return f"{size}{connection}outlet".strip()
    return f"{size}{feature.kind.replace('_', ' ')}".strip()


def generate_diagram(
    path: Path,
    spool: Spool,
    geometry: SpoolGeometry,
    issues: list[ValidationIssue],
) -> None:
    total = float(geometry.total_length.as_fraction())
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    try:
        axis.plot([0, total], [0, 0], color="#293241", linewidth=10, solid_capstyle="butt")
        labelled = 0
        for feature in spool.features:
            position_value = geometry.positions.get(feature.id)
            if position_value is None:
                continue
            position = float(position_value.as_fraction())
            if feature.kind == "outlet":
                direction = -1 if feature.orientation == "down" else 1
                axis.plot([position, position], [0, direction * 0.55], color="#d1495b", linewidth=5)
            # Alternate by label order, not by position value, so that adjacent
            # labels never collide on evenly spaced features.
            axis.annotate(
                f"{_feature_label(feature)}\n{position_value.display}",
                xy=(position, 0),
                xytext=(0, 25 if labelled % 2 else -42),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
            labelled += 1
        axis.annotate(
            f"TOTAL {geometry.total_length.display}",
            xy=(total / 2, 0),
            xytext=(0, 55),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            weight="bold",
        )
        warnings = [issue.message for issue in issues if issue.spool_id == spool.id]
        if warnings:
            axis.text(0, -0.9, "\n".join(warnings), color="#9c2c2c", fontsize=8, va="top")
        axis.text(
            0.5,
            0.02,
            REVIEW_LABEL,
            transform=figure.transFigure,
            ha="center",
            color="#b00020",
            fontsize=12,
            weight="bold",
        )
        axis.set_xlim(-max(total * 0.05, 1), total * 1.05)
        axis.set_ylim(-1.2, 1.2)
        axis.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, metadata={"Title": REVIEW_LABEL})
    finally:
        plt.close(figure)
