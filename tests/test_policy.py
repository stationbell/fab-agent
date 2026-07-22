import pytest

from fab_agent.application.policy import set_human_field
from fab_agent.errors import ModelError


def test_human_answer_only_updates_allowlisted_raw_field() -> None:
    data = {"spools": [{"id": "spool-001", "material_raw": None}]}
    set_human_field(data, "spools.spool-001.material_raw", "carbon steel")
    assert data["spools"][0]["material_raw"] == "carbon steel"


def test_human_answer_rejects_derived_field() -> None:
    with pytest.raises(ModelError, match="not allowlisted"):
        set_human_field(
            {"spools": [{"id": "spool-001"}]},
            "spools.spool-001.normalized_total",
            "100",
        )
