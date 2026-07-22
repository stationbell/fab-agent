from __future__ import annotations

from pathlib import Path

import pytest

from fab_agent.config import OllamaConfig, load_config
from fab_agent.domain.design import FabricationDesign
from fab_agent.errors import ConfigurationError
from fab_agent.infrastructure.catalogs import load_catalogs
from fab_agent.infrastructure.serialization import read_toml, write_toml


def test_relative_paths_resolve_from_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "settings" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        '[output]\nroot = "output"\n\n[catalogs]\nroot = "../catalogs"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.output.root == (config_path.parent / "output").resolve()
    assert config.catalogs.root == (config_path.parent / "../catalogs").resolve()


def test_environment_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[ollama]\nbase_url = "http://file-value"\n')
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://environment-value")
    assert load_config(config_path).ollama.base_url == "http://environment-value"


def test_ollama_connection_selects_a_safe_default_endpoint() -> None:
    assert OllamaConfig(connection="local").resolved_base_url == "http://localhost:11434"
    assert OllamaConfig(connection="cloud").resolved_base_url == "https://ollama.com"


def test_ollama_connection_can_be_overridden_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[ollama]\nconnection = "local"\n')
    monkeypatch.setenv("FAB_AGENT_OLLAMA_CONNECTION", "cloud")
    assert load_config(config_path).ollama.connection == "cloud"


@pytest.mark.parametrize(
    "base_url",
    [
        "ollama.com",
        "https://user:secret@ollama.com",
        "https://ollama.com/api",
        "https://ollama.com?token=secret",
    ],
)
def test_ollama_base_url_must_be_a_credential_free_origin(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaConfig(base_url=base_url)


def test_ollama_cloud_rejects_plaintext_custom_endpoint() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OllamaConfig(connection="cloud", base_url="http://ollama.test")


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize("invalid_value", ["four and a half", "1/0"])
def test_unreadable_catalog_geometry_is_reported_against_the_catalog(
    tmp_path: Path, invalid_value: str
) -> None:
    """A bad catalog value must not surface later as an invalid source dimension."""

    (tmp_path / "pipe.toml").write_text(
        "schema_version = 1\n\n"
        "[[pipes]]\n"
        'key = "broken"\n'
        'nominal_size = "4"\n'
        'schedule = "10"\n'
        'material = "demo carbon steel"\n'
        f'outside_diameter_in = "{invalid_value}"\n'
        'wall_thickness_in = "1/8"\n',
        encoding="utf-8",
    )
    (tmp_path / "components.toml").write_text(
        "schema_version = 1\ncomponents = []\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="Invalid catalog schema"):
        load_catalogs(tmp_path)


def test_catalog_nominal_size_may_not_be_written_in_feet(tmp_path: Path) -> None:
    (tmp_path / "pipe.toml").write_text("schema_version = 1\npipes = []\n", encoding="utf-8")
    (tmp_path / "components.toml").write_text(
        "schema_version = 1\n\n"
        "[[components]]\n"
        'key = "wrong-units"\n'
        'kind = "coupling"\n'
        'nominal_size = "4\'"\n'
        'outside_diameter_in = "5 1/4"\n'
        'length_in = "3"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Invalid catalog schema"):
        load_catalogs(tmp_path)


def test_design_toml_round_trip(tmp_path: Path, valid_design: FabricationDesign) -> None:
    path = tmp_path / "design.toml"
    write_toml(path, valid_design.model_dump(mode="json", exclude_none=True))
    assert FabricationDesign.model_validate(read_toml(path)) == valid_design
