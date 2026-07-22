from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from fab_agent.config import OllamaConfig
from fab_agent.errors import ModelError
from fab_agent.models.ollama import OllamaModelClient, _inline_schema_references
from fab_agent.models.schemas import PipeSpoolObservation

SUBMIT_TOOL = "submit_structured_response"


def _observation_arguments() -> dict[str, Any]:
    return {
        "spools": [],
        "observed_components": [],
        "notes": [],
        "uncertainties": [],
    }


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "sketch.jpg"
    path.write_bytes(b"image bytes")
    return path


def _tool_completion(arguments: dict[str, Any] | str, *, name: str = SUBMIT_TOOL) -> httpx.Response:
    encoded = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": name, "arguments": encoded},
                            }
                        ],
                    }
                }
            ]
        },
    )


def test_image_schema_references_are_fully_inlined() -> None:
    schema = _inline_schema_references(PipeSpoolObservation.model_json_schema())
    encoded = json.dumps(schema)
    assert "$defs" not in encoded
    assert "$ref" not in encoded
    assert schema["properties"]["spools"]["items"]["type"] == "object"


def test_extract_image_forces_one_submission_tool_and_retries_invalid_output(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "no tool"}}]})
        return _tool_completion(_observation_arguments())

    http_client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        retries=1,
        client=http_client,
    )
    result = asyncio.run(
        client.extract_image(
            prompt="read",
            response_type=PipeSpoolObservation,
            image_path=_image(tmp_path),
        )
    )
    asyncio.run(client.aclose())

    assert result.spools == []
    assert payloads[0]["stream"] is False
    assert payloads[0]["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in payloads[0]["tools"]] == [SUBMIT_TOOL]
    assert "never add undeclared properties" in payloads[0]["messages"][0]["content"]
    assert "prior tool arguments were invalid" in payloads[1]["messages"][-1]["content"]


def test_extract_image_fails_after_retry_budget(tmp_path: Path) -> None:
    http_client = httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"choices": [{"message": {}}]})
        ),
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        retries=1,
        client=http_client,
    )
    with pytest.raises(ModelError, match="Invalid structured response"):
        asyncio.run(
            client.extract_image(
                prompt="read",
                response_type=PipeSpoolObservation,
                image_path=_image(tmp_path),
            )
        )
    asyncio.run(client.aclose())


def test_health_uses_openai_models_endpoint_and_checks_only_vision_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"id": "vision"}]})

    http_client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        client=http_client,
    )
    health = asyncio.run(client.health())
    asyncio.run(client.aclose())

    assert health["connection"] == "local"
    assert health["vision_model_available"] is True
    assert "orchestrator_model_available" not in health


def test_cloud_requests_use_openai_api_and_api_key_without_storing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "private-test-key"
    monkeypatch.setenv("CUSTOM_OLLAMA_KEY", api_key)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {api_key}"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3.5:397b"}]})
        assert request.url.path == "/v1/chat/completions"
        return _tool_completion(_observation_arguments())

    config = OllamaConfig(
        connection="cloud",
        api_key_env="CUSTOM_OLLAMA_KEY",
        vision_model="qwen3.5:397b",
    )
    http_client = httpx.AsyncClient(
        base_url=config.resolved_base_url, transport=httpx.MockTransport(handler)
    )
    client = OllamaModelClient(config, client=http_client)

    health = asyncio.run(client.health())
    result = asyncio.run(
        client.extract_image(
            prompt="read",
            response_type=PipeSpoolObservation,
            image_path=_image(tmp_path),
        )
    )
    asyncio.run(client.aclose())

    assert health["base_url"] == "https://ollama.com"
    assert health["vision_model_available"] is True
    assert result.spools == []
    assert api_key not in repr(config)
    assert [request.url.path for request in requests] == [
        "/v1/models",
        "/v1/chat/completions",
    ]


def test_vision_request_uses_openai_image_content(tmp_path: Path) -> None:
    image = tmp_path / "sketch.png"
    image.write_bytes(b"image bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][-1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1] == {"type": "text", "text": "read this"}
        return _tool_completion(_observation_arguments())

    http_client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        client=http_client,
    )
    asyncio.run(
        client.extract_image(
            prompt="read this",
            response_type=PipeSpoolObservation,
            image_path=image,
        )
    )
    asyncio.run(client.aclose())


def test_image_response_records_and_removes_redundant_design_type(tmp_path: Path) -> None:
    arguments = _observation_arguments()
    arguments["type"] = "OBJECT"
    http_client = httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(lambda _: _tool_completion(arguments)),
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        retries=0,
        client=http_client,
    )

    result = asyncio.run(
        client.extract_image(
            prompt="read",
            response_type=PipeSpoolObservation,
            image_path=_image(tmp_path),
        )
    )
    asyncio.run(client.aclose())

    assert result.uncertainties == [
        "Ollama adapter ignored redundant non-engineering model metadata type='OBJECT'."
    ]


def test_image_response_wraps_known_singleton_array_fields(tmp_path: Path) -> None:
    arguments = _observation_arguments()
    arguments["spools"] = {"id_raw": "S1"}
    arguments["observed_components"] = {
        "description_raw": '2 - 4" couplings',
        "quantity": 2,
    }
    http_client = httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(lambda _: _tool_completion(arguments)),
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        retries=0,
        client=http_client,
    )

    result = asyncio.run(
        client.extract_image(
            prompt="read",
            response_type=PipeSpoolObservation,
            image_path=_image(tmp_path),
        )
    )
    asyncio.run(client.aclose())

    assert [spool.id_raw for spool in result.spools] == ["S1"]
    assert result.observed_components[0].quantity == 2
    assert result.uncertainties == [
        "Ollama adapter wrapped singleton 'spools' object as a one-item list.",
        "Ollama adapter wrapped singleton 'observed_components' object as a one-item list.",
    ]


def test_image_response_decodes_stringified_known_array_fields(tmp_path: Path) -> None:
    arguments = _observation_arguments()
    arguments["spools"] = json.dumps({"id_raw": "S1"})
    arguments["observed_components"] = json.dumps(
        [{"description_raw": '2 - 4" couplings', "quantity": 2}]
    )
    http_client = httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(lambda _: _tool_completion(arguments)),
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        retries=0,
        client=http_client,
    )

    result = asyncio.run(
        client.extract_image(
            prompt="read",
            response_type=PipeSpoolObservation,
            image_path=_image(tmp_path),
        )
    )
    asyncio.run(client.aclose())

    assert [spool.id_raw for spool in result.spools] == ["S1"]
    assert result.observed_components[0].quantity == 2
    assert result.uncertainties == [
        "Ollama adapter decoded JSON string in 'spools' array field.",
        "Ollama adapter wrapped singleton 'spools' object as a one-item list.",
        "Ollama adapter decoded JSON string in 'observed_components' array field.",
    ]


def test_image_response_still_rejects_every_other_extra_field(tmp_path: Path) -> None:
    arguments = _observation_arguments()
    arguments.update({"type": "OBJECT", "unexpected": "unsafe"})
    http_client = httpx.AsyncClient(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(lambda _: _tool_completion(arguments)),
    )
    client = OllamaModelClient(
        OllamaConfig(base_url="http://ollama.test", vision_model="vision"),
        retries=0,
        client=http_client,
    )

    with pytest.raises(ModelError, match=r"unexpected: Extra inputs"):
        asyncio.run(
            client.extract_image(
                prompt="read",
                response_type=PipeSpoolObservation,
                image_path=_image(tmp_path),
            )
        )
    asyncio.run(client.aclose())


def test_cloud_request_fails_before_network_when_api_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_OLLAMA_KEY", raising=False)
    network_called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal network_called
        network_called = True
        return httpx.Response(500)

    config = OllamaConfig(
        connection="cloud",
        api_key_env="MISSING_OLLAMA_KEY",
        vision_model="qwen3.5:397b",
    )
    client = OllamaModelClient(
        config,
        client=httpx.AsyncClient(
            base_url=config.resolved_base_url, transport=httpx.MockTransport(handler)
        ),
    )

    with pytest.raises(ModelError, match="MISSING_OLLAMA_KEY"):
        asyncio.run(client.health())
    asyncio.run(client.aclose())
    assert network_called is False


def test_cloud_authentication_error_names_variable_not_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "rejected-private-key"
    monkeypatch.setenv("REJECTED_OLLAMA_KEY", api_key)
    config = OllamaConfig(
        connection="cloud",
        api_key_env="REJECTED_OLLAMA_KEY",
        vision_model="qwen3.5:397b",
    )
    client = OllamaModelClient(
        config,
        client=httpx.AsyncClient(
            base_url=config.resolved_base_url,
            transport=httpx.MockTransport(lambda _: httpx.Response(401)),
        ),
    )

    with pytest.raises(ModelError, match="REJECTED_OLLAMA_KEY") as caught:
        asyncio.run(client.health())
    asyncio.run(client.aclose())
    assert api_key not in str(caught.value)
