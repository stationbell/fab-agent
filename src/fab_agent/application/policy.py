"""Allowlisted human corrections for persisted source fields."""

from __future__ import annotations

import re
from typing import Any

from fab_agent.errors import ModelError

_RAW_FIELD_PATTERN = re.compile(
    r"^spools\.[A-Za-z0-9_-]+\."
    r"(?:nominal_size_raw|schedule_raw|material_raw|stated_total_length_raw)$"
)


def set_human_field(design_data: dict[str, Any], target_field: str, answer: str) -> None:
    if not _RAW_FIELD_PATTERN.fullmatch(target_field):
        raise ModelError("Pending human-answer field is not allowlisted")
    _, spool_id, field_name = target_field.split(".")
    for spool in design_data.get("spools", []):
        if spool.get("id") == spool_id:
            spool[field_name] = answer
            return
    raise ModelError(f"Pending question references missing spool {spool_id}")
