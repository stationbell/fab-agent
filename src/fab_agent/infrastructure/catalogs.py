"""Human-editable deterministic geometry catalogs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict

from fab_agent.domain.dimensions import parse_dimension, parse_nominal_size
from fab_agent.errors import ConfigurationError


class CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _exact_length(value: str) -> str:
    parse_dimension(value)
    return value


def _exact_nominal_size(value: str) -> str:
    parse_nominal_size(value)
    return value


# Catalog geometry is parsed once at load so that an unreadable catalog value is
# reported against the catalog file rather than misattributed to the sketch.
LengthValue = Annotated[str, AfterValidator(_exact_length)]
NominalSizeValue = Annotated[str, AfterValidator(_exact_nominal_size)]


class PipeCatalogEntry(CatalogModel):
    key: str
    nominal_size: NominalSizeValue
    schedule: str
    material: str
    outside_diameter_in: LengthValue
    wall_thickness_in: LengthValue
    demo_only: bool = False


class ComponentCatalogEntry(CatalogModel):
    key: str
    kind: str
    nominal_size: NominalSizeValue
    outside_diameter_in: LengthValue
    length_in: LengthValue
    demo_only: bool = False


class CatalogBundle(CatalogModel):
    pipes: tuple[PipeCatalogEntry, ...]
    components: tuple[ComponentCatalogEntry, ...]

    def find_pipe(
        self, nominal_size_raw: str, schedule: str, material: str, *, allow_demo: bool
    ) -> PipeCatalogEntry | None:
        target_nps = parse_nominal_size(nominal_size_raw).inches
        for entry in self.pipes:
            if entry.demo_only and not allow_demo:
                continue
            if (
                parse_nominal_size(entry.nominal_size).inches == target_nps
                and _normalize_schedule(entry.schedule) == _normalize_schedule(schedule)
                and entry.material.casefold() == material.casefold()
            ):
                return entry
        return None

    def find_component(
        self, kind: str, nominal_size_raw: str, *, allow_demo: bool
    ) -> ComponentCatalogEntry | None:
        target_nps = parse_nominal_size(nominal_size_raw).inches
        for entry in self.components:
            if entry.demo_only and not allow_demo:
                continue
            if entry.kind == kind and parse_nominal_size(entry.nominal_size).inches == target_nps:
                return entry
        return None


def _normalize_schedule(value: str) -> str:
    """Normalize common source forms such as ``10``, ``SCH 10``, and ``sch. 10``."""

    normalized = value.strip().casefold()
    normalized = re.sub(r"^(?:schedule|sch)\.?\s*", "", normalized)
    return re.sub(r"\s+", "", normalized)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Cannot load catalog {path}: {exc}") from exc


def load_catalogs(root: Path) -> CatalogBundle:
    pipe_path = root / "pipe.toml"
    component_path = root / "components.toml"
    pipe_data = _load_toml(pipe_path)
    component_data = _load_toml(component_path)
    try:
        return CatalogBundle(
            pipes=tuple(PipeCatalogEntry.model_validate(item) for item in pipe_data["pipes"]),
            components=tuple(
                ComponentCatalogEntry.model_validate(item) for item in component_data["components"]
            ),
        )
    except (KeyError, ValueError) as exc:
        raise ConfigurationError(f"Invalid catalog schema under {root}: {exc}") from exc
