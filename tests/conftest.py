from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel

from fab_agent.config import (
    AppConfig,
    CadConfig,
    CatalogsConfig,
    ImagesConfig,
    OllamaConfig,
    OutputConfig,
    WorkflowConfig,
)
from fab_agent.domain.design import FabricationDesign, Feature, Segment, Spool
from fab_agent.infrastructure.catalogs import CatalogBundle, load_catalogs

ROOT = Path(__file__).parents[1]


@pytest.fixture
def catalogs() -> CatalogBundle:
    return load_catalogs(ROOT / "catalogs")


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ollama=OllamaConfig(vision_model="fake-vision"),
        workflow=WorkflowConfig(),
        images=ImagesConfig(minimum_bytes=1, maximum_bytes=10_000_000),
        output=OutputConfig(root=tmp_path / "runs"),
        catalogs=CatalogsConfig(root=ROOT / "catalogs", allow_demo_entries=True),
        cad=CadConfig(enabled=False),
    )


@pytest.fixture
def input_image(tmp_path: Path) -> Path:
    path = tmp_path / "sketch.png"
    Image.new("RGB", (640, 480), "white").save(path)
    return path


@pytest.fixture
def valid_design() -> FabricationDesign:
    return FabricationDesign(
        project_reference_raw="Synthetic test",
        spools=[
            Spool(
                id="spool-001",
                nominal_size_raw='4"',
                schedule_raw="10",
                material_raw="demo carbon steel",
                stated_total_length_raw="8 ft",
                features=[
                    Feature(
                        id="left",
                        kind="coupling",
                        nominal_size_raw='4"',
                    ),
                    Feature(
                        id="outlet",
                        kind="outlet",
                        nominal_size_raw='1 1/4"',
                        connection_type="threaded",
                        orientation="up",
                    ),
                    Feature(
                        id="right",
                        kind="coupling",
                        nominal_size_raw='4"',
                    ),
                ],
                segments=[
                    Segment(
                        id="s1",
                        from_feature_id="left",
                        to_feature_id="outlet",
                        length_raw="3 ft",
                    ),
                    Segment(
                        id="s2",
                        from_feature_id="outlet",
                        to_feature_id="right",
                        length_raw="5 ft",
                    ),
                ],
            )
        ],
    )


class FakeModel:
    vision_model = "fake-vision"

    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self.responses = responses
        self.calls = 0

    async def extract_image(
        self,
        *,
        prompt: str,
        response_type: type[BaseModel],
        image_path: Path,
    ) -> BaseModel:
        del prompt, image_path
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, response_type)
        return response

    async def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def aclose(self) -> None:
        return None
