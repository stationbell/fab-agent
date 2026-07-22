"""Configuration loading with explicit and predictable precedence."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from fab_agent.errors import ConfigurationError


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OllamaConfig(_ConfigModel):
    connection: Literal["local", "cloud"] = "local"
    base_url: str | None = None
    api_key_env: str = Field(
        default="OLLAMA_API_KEY",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    vision_model: str = "CHANGE_ME"
    request_timeout_seconds: PositiveInt = 180

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain credentials")
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an origin without a path, query, or fragment")
        return normalized

    @property
    def resolved_base_url(self) -> str:
        if self.base_url is not None:
            return self.base_url
        if self.connection == "cloud":
            return "https://ollama.com"
        return "http://localhost:11434"

    @model_validator(mode="after")
    def cloud_requires_https(self) -> Self:
        if self.connection == "cloud" and urlsplit(self.resolved_base_url).scheme != "https":
            raise ValueError("cloud connection requires an HTTPS base_url")
        return self


class WorkflowConfig(_ConfigModel):
    max_human_questions: PositiveInt = 1
    structured_output_retry_count: int = Field(default=1, ge=0, le=3)


class ImagesConfig(_ConfigModel):
    max_edge_pixels: PositiveInt = 2000
    minimum_bytes: PositiveInt = 15_360
    maximum_bytes: PositiveInt = 20_971_520
    allowed_formats: tuple[str, ...] = ("jpeg", "png", "gif", "webp", "heic")

    @field_validator("maximum_bytes")
    @classmethod
    def maximum_exceeds_minimum(cls, value: int, info: Any) -> int:
        minimum = info.data.get("minimum_bytes", 0)
        if value <= minimum:
            raise ValueError("maximum_bytes must exceed minimum_bytes")
        return value


class OutputConfig(_ConfigModel):
    root: Path = Path("runs")


class CatalogsConfig(_ConfigModel):
    root: Path = Path("catalogs")
    allow_demo_entries: bool = False


class ValidationConfig(_ConfigModel):
    dimensional_tolerance_in: str = "1/16"


class CadConfig(_ConfigModel):
    enabled: bool = True


class AppConfig(_ConfigModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    catalogs: CatalogsConfig = Field(default_factory=CatalogsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cad: CadConfig = Field(default_factory=CadConfig)
    config_path: Path | None = None


_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "FAB_AGENT_OLLAMA_CONNECTION": ("ollama", "connection"),
    "OLLAMA_BASE_URL": ("ollama", "base_url"),
    "FAB_AGENT_OLLAMA_API_KEY_ENV": ("ollama", "api_key_env"),
    "FAB_AGENT_VISION_MODEL": ("ollama", "vision_model"),
    "FAB_AGENT_OUTPUT_ROOT": ("output", "root"),
    "FAB_AGENT_CATALOG_ROOT": ("catalogs", "root"),
}


def _resolve_paths(data: dict[str, Any], base: Path) -> None:
    for section, key in (("output", "root"), ("catalogs", "root")):
        raw = data.get(section, {}).get(key)
        if raw is not None:
            path = Path(raw)
            data[section][key] = path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path | None = None) -> AppConfig:
    """Load config using CLI path, FAB_AGENT_CONFIG, then ./config.toml."""

    configured = path or (Path(value) if (value := os.getenv("FAB_AGENT_CONFIG")) else None)
    candidate = configured or Path("config.toml")
    data: dict[str, Any] = {}
    config_path: Path | None = None
    if candidate.exists():
        config_path = candidate.resolve()
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"Cannot load config {config_path}: {exc}") from exc
    elif configured is not None:
        raise ConfigurationError(f"Config file does not exist: {candidate}")

    for env_name, (section, key) in _ENV_OVERRIDES.items():
        if value := os.getenv(env_name):
            data.setdefault(section, {})[key] = value

    base = config_path.parent if config_path else Path.cwd()
    _resolve_paths(data, base)
    data["config_path"] = config_path
    try:
        return AppConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
