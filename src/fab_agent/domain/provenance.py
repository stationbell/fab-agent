"""Field-level source provenance."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from fab_agent.domain.design import DomainModel, FabricationDesign


class ProvenanceEntry(DomainModel):
    field_path: str
    source_type: Literal["image", "human_answer", "manual_edit"]
    raw_text: str
    recorded_at: datetime
    model: str | None = None


class ProvenanceDocument(DomainModel):
    schema_version: Literal[1] = 1
    entries: list[ProvenanceEntry] = Field(default_factory=list)


def changed_scalar_fields(before: dict[str, Any], after: dict[str, Any]) -> dict[str, str]:
    """Return human-readable scalar changes for manual-edit provenance."""

    before_flat = _flatten(before)
    after_flat = _flatten(after)
    return {
        path: str(after_flat.get(path, "<removed>"))
        for path in sorted(before_flat.keys() | after_flat.keys())
        if before_flat.get(path, "<missing>") != after_flat.get(path, "<missing>")
    }


def observed_source_fields(design: FabricationDesign) -> dict[str, str]:
    """Enumerate every persisted model-observed scalar that needs provenance."""

    flattened = _flatten(design.model_dump(mode="python", exclude_none=True))
    excluded_names = {"schema_version", "id"}
    return {
        path: str(value)
        for path, value in flattened.items()
        if path.rsplit(".", 1)[-1] not in excluded_names
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flattened.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            identifier = item.get("id") if isinstance(item, dict) else None
            key = str(identifier) if identifier else str(index)
            flattened.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        flattened[prefix] = value
    return flattened
