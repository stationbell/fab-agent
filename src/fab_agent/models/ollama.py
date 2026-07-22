"""Small OpenAI-compatible client for local Ollama and Ollama Cloud."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from fab_agent.config import OllamaConfig
from fab_agent.errors import ModelError
from fab_agent.models.schemas import PipeSpoolObservation

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_SUBMIT_TOOL = "submit_structured_response"


def _inline_schema_references(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local $defs references for models that do not reliably resolve them."""

    source = deepcopy(schema)
    definitions = source.pop("$defs", {})

    def resolve(value: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [resolve(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        if reference := value.get("$ref"):
            prefix = "#/$defs/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise ValueError(f"Unsupported JSON Schema reference: {reference}")
            name = reference.removeprefix(prefix)
            if name in stack:
                raise ValueError(f"Recursive JSON Schema reference: {name}")
            target = definitions.get(name)
            if target is None:
                raise ValueError(f"Missing JSON Schema definition: {name}")
            merged = {**target, **{key: item for key, item in value.items() if key != "$ref"}}
            return resolve(merged, (*stack, name))
        return {key: resolve(item, stack) for key, item in value.items()}

    result = resolve(source)
    if not isinstance(result, dict):
        raise ValueError("JSON Schema must be an object")
    return result


def _validation_hint(error: Exception | None) -> str:
    if not isinstance(error, ValidationError):
        return ""
    details = error.errors(include_url=False, include_context=False, include_input=True)[:6]
    return "; ".join(
        f"{'.'.join(map(str, item['loc']))}: {item['msg']} "
        f"(received {type(item.get('input')).__name__})"
        for item in details
    )


def _error_summary(error: Exception | None) -> str:
    if isinstance(error, ValidationError):
        hint = _validation_hint(error)
        return f"{error.error_count()} schema validation errors; first errors: {hint}"
    return str(error)


def _normalize_image_arguments(arguments: Any, response_type: type[BaseModel]) -> Any:
    """Normalize a few schema-safe shapes emitted by some vision models.

    Some vision models add a string discriminator at ``type`` even though the
    selected schema has no discriminator or emit an object for a known array field
    when there is one item. These changes cannot alter fabrication values, and every
    normalization is recorded as an uncertainty. No undeclared properties are accepted.
    """

    if not isinstance(arguments, dict):
        return arguments
    normalized = deepcopy(arguments)
    changes: list[str] = []
    if response_type is PipeSpoolObservation and isinstance(normalized.get("type"), str):
        ignored_type = normalized.pop("type")[:80]
        changes.append(
            "Ollama adapter ignored redundant non-engineering model metadata "
            f"type={ignored_type!r}."
        )
    if response_type is PipeSpoolObservation:
        for field_name in ("spools", "observed_components"):
            field_value = normalized.get(field_name)
            if isinstance(field_value, str):
                try:
                    decoded = json.loads(field_value)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, (dict, list)):
                    normalized[field_name] = decoded
                    changes.append(
                        f"Ollama adapter decoded JSON string in {field_name!r} array field."
                    )
            if isinstance(normalized.get(field_name), dict):
                normalized[field_name] = [normalized[field_name]]
                changes.append(
                    f"Ollama adapter wrapped singleton {field_name!r} object as a one-item list."
                )
    if not changes:
        return arguments

    uncertainties = normalized.setdefault("uncertainties", [])
    if isinstance(uncertainties, list):
        uncertainties.extend(changes)
    return normalized


class OllamaModelClient:
    def __init__(
        self,
        config: OllamaConfig,
        *,
        retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.retries = retries
        self._client = client or httpx.AsyncClient(
            base_url=config.resolved_base_url,
            timeout=float(config.request_timeout_seconds),
        )

    def _auth_headers(self) -> dict[str, str]:
        if self.config.connection == "local":
            return {}
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ModelError(
                f"Ollama Cloud requires environment variable {self.config.api_key_env}"
            )
        return {"Authorization": f"Bearer {api_key}"}

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if self.config.connection == "cloud" and response.status_code in {401, 403}:
                raise ModelError(
                    "Ollama Cloud authentication failed; "
                    f"check environment variable {self.config.api_key_env}"
                ) from exc
            raise

    @property
    def vision_model(self) -> str:
        return self.config.vision_model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/v1/models", headers=self._auth_headers())
            self._raise_for_status(response)
            data = response.json()
        except ModelError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError(
                f"Ollama is unavailable at {self.config.resolved_base_url}: {exc}"
            ) from exc
        available = {item.get("id", "") for item in data.get("data", [])}
        return {
            "connection": self.config.connection,
            "base_url": self.config.resolved_base_url,
            "available_models": sorted(available),
            "vision_model_available": self.vision_model in available,
        }

    async def extract_image(
        self,
        *,
        prompt: str,
        response_type: type[ResponseT],
        image_path: Path,
    ) -> ResponseT:
        if not image_path.is_file():
            raise ModelError(f"Model image does not exist: {image_path}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        request_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        schema = _inline_schema_references(response_type.model_json_schema())
        schema.setdefault("type", "object")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": _SUBMIT_TOOL,
                    "description": "Submit the requested response for local validation.",
                    "parameters": schema,
                },
            }
        ]
        request_messages = [
            {
                "role": "system",
                "content": (
                    f"You must call {_SUBMIT_TOOL} exactly once. "
                    "Do not answer with ordinary text. Tool arguments must match the supplied "
                    "schema exactly; never add undeclared properties."
                ),
            },
            *request_messages,
        ]
        payload: dict[str, Any] = {
            "model": self.vision_model,
            "messages": request_messages,
            "stream": False,
            "temperature": 0,
            "tools": tools,
            "tool_choice": "required",
        }
        timeout_seconds = self.config.request_timeout_seconds
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if attempt:
                    hint = _validation_hint(last_error)
                    payload["messages"] = [
                        *request_messages,
                        {
                            "role": "user",
                            "content": (
                                "The prior tool arguments were invalid. Call the tool again."
                                + (f" Fix these fields: {hint}" if hint else "")
                            ),
                        },
                    ]
                response = await self._client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=float(timeout_seconds),
                )
                self._raise_for_status(response)
                tool_calls = response.json()["choices"][0]["message"]["tool_calls"]
                if len(tool_calls) != 1:
                    raise ValueError("model must return exactly one tool call")
                function = tool_calls[0]["function"]
                arguments = function["arguments"]
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                if function["name"] != _SUBMIT_TOOL:
                    raise ValueError(f"model called unexpected tool {function['name']}")
                if response_type is PipeSpoolObservation:
                    parsed = _normalize_image_arguments(parsed, response_type)
                return response_type.model_validate(parsed)
            except ModelError:
                raise
            except httpx.TimeoutException as exc:
                last_error = exc
                break
            except (
                httpx.HTTPError,
                KeyError,
                TypeError,
                ValueError,
                ValidationError,
            ) as exc:
                last_error = exc
        if isinstance(last_error, httpx.TimeoutException):
            raise ModelError(
                f"Timed out waiting for model {self.vision_model} after {timeout_seconds} seconds"
            ) from last_error
        raise ModelError(
            "Invalid structured response from model "
            f"{self.vision_model}: {_error_summary(last_error)}"
        ) from last_error
