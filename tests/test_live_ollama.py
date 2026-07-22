from __future__ import annotations

import asyncio
import os

import pytest

from fab_agent.config import load_config
from fab_agent.infrastructure.models_factory import build_model_client


@pytest.mark.live_ollama
@pytest.mark.skipif(
    os.getenv("FAB_AGENT_RUN_LIVE_OLLAMA") != "1",
    reason="set FAB_AGENT_RUN_LIVE_OLLAMA=1 to exercise configured Ollama",
)
def test_configured_ollama_vision_model_is_available() -> None:
    client = build_model_client(load_config())
    health = asyncio.run(client.health())
    asyncio.run(client.aclose())
    assert health["vision_model_available"]
