"""Deterministic conversion from flat visual observations to the domain graph."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fab_agent.domain.design import FabricationDesign, Feature, Segment, Spool
from fab_agent.models.schemas import PipeSpoolObservation

_GENERIC_MATERIAL_VALUES = {
    "n/a",
    "not shown",
    "pipe",
    "unknown",
    "unspecified",
}

_EXPLICIT_COMPONENT_KINDS = (
    (re.compile(r"\bcouplings?\b", re.IGNORECASE), "coupling"),
    (re.compile(r"\bcaps?\b", re.IGNORECASE), "cap"),
    (re.compile(r"\bthreaded\s+outlets?\b", re.IGNORECASE), "threaded_outlet"),
    (re.compile(r"\bgrooved\s+outlets?\b", re.IGNORECASE), "grooved_outlet"),
    (re.compile(r"\boutlets?\b", re.IGNORECASE), "outlet"),
)


@dataclass(frozen=True, slots=True)
class ConvertedPipeSpool:
    design: FabricationDesign
    uncertainties: list[str]


def _material_or_none(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if value.strip().casefold() in _GENERIC_MATERIAL_VALUES:
        return None, f"Ignored non-material placeholder {value!r}; pipe material remains missing."
    return value, None


def _explicit_component_kind(description: str) -> str | None:
    for pattern, kind in _EXPLICIT_COMPONENT_KINDS:
        if pattern.search(description):
            return kind
    return None


def convert_observation(observation: PipeSpoolObservation) -> ConvertedPipeSpool:
    """Assign identifiers and graph links without asking the model to author topology."""

    uncertainties = list(observation.uncertainties)
    spools: list[Spool] = []
    used_spool_ids: set[str] = set()
    for spool_index, observed in enumerate(observation.spools, start=1):
        candidate_id = (observed.id_raw or f"S{spool_index}").strip() or f"S{spool_index}"
        spool_id = candidate_id
        suffix = 2
        while spool_id in used_spool_ids:
            spool_id = f"{candidate_id}-{suffix}"
            suffix += 1
        used_spool_ids.add(spool_id)

        material, material_warning = _material_or_none(observed.material_raw)
        if material_warning:
            uncertainties.append(f"Spool {spool_id}: {material_warning}")

        features: list[Feature] = []
        last_feature_index = len(observed.physical_features_left_to_right) - 1
        for feature_index, observed_feature in enumerate(observed.physical_features_left_to_right):
            feature_data = observed_feature.model_dump(exclude_none=True)
            if observed_feature.kind == "unknown_end" and feature_index in {
                0,
                last_feature_index,
            }:
                boundary_kind = "start" if feature_index == 0 else "end"
                feature_data["kind"] = boundary_kind
                uncertainties.append(
                    f"Spool {spool_id}: normalized boundary feature {feature_index + 1} "
                    f"from unknown_end to {boundary_kind}; source labels were preserved."
                )
            features.append(Feature(id=f"f{feature_index + 1}", **feature_data))
        expected_segments = max(0, len(features) - 1)
        if len(observed.segment_lengths_left_to_right) != expected_segments:
            uncertainties.append(
                f"Spool {spool_id}: observed {len(observed.segment_lengths_left_to_right)} "
                f"segment dimensions for {len(features)} physical features; expected "
                f"{expected_segments}."
            )
        segments = [
            Segment(
                id=f"seg{index + 1}",
                from_feature_id=features[index].id,
                to_feature_id=features[index + 1].id,
                length_raw=length,
            )
            for index, length in enumerate(
                observed.segment_lengths_left_to_right[:expected_segments]
            )
        ]
        spools.append(
            Spool(
                id=spool_id,
                nominal_size_raw=observed.nominal_size_raw,
                schedule_raw=observed.schedule_raw,
                material_raw=material,
                stated_total_length_raw=observed.stated_total_length_raw,
                features=features,
                segments=segments,
                orientation_legend=observed.orientation_legend,
                notes=observed.notes,
            )
        )

    observed_components = []
    for component in observation.observed_components:
        if component.kind:
            observed_components.append(component)
            continue
        explicit_kind = _explicit_component_kind(component.description_raw)
        if explicit_kind is None:
            observed_components.append(component)
            continue
        observed_components.append(component.model_copy(update={"kind": explicit_kind}))
        uncertainties.append(
            f"Recovered observed component kind {explicit_kind!r} from explicit parts-list "
            f"text {component.description_raw!r}."
        )

    return ConvertedPipeSpool(
        design=FabricationDesign(
            project_reference_raw=observation.project_reference_raw,
            spools=spools,
            observed_components=observed_components,
            notes=observation.notes,
        ),
        uncertainties=uncertainties,
    )
