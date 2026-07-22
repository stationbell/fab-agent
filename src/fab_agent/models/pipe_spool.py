"""Deterministic conversion from flat visual observations to the domain graph."""

from __future__ import annotations

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

        features = [
            Feature(id=f"f{index}", **feature.model_dump(exclude_none=True))
            for index, feature in enumerate(
                observed.physical_features_left_to_right,
                start=1,
            )
        ]
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

    return ConvertedPipeSpool(
        design=FabricationDesign(
            project_reference_raw=observation.project_reference_raw,
            spools=spools,
            observed_components=observation.observed_components,
            notes=observation.notes,
        ),
        uncertainties=uncertainties,
    )
