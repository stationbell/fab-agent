"""Thin, human-friendly Typer adapter."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, NoReturn

import typer

from fab_agent.api import resume_fab_agent, run_fab_agent
from fab_agent.application.runner import Dependencies, rebuild_run
from fab_agent.config import load_config
from fab_agent.domain.results import FabRequest, FabResult
from fab_agent.errors import FabAgentError
from fab_agent.infrastructure.catalogs import load_catalogs
from fab_agent.infrastructure.filesystem import FilesystemRunStore
from fab_agent.infrastructure.models_factory import build_model_client
from fab_agent.infrastructure.runtime import SecureIdGenerator, SystemClock
from fab_agent.infrastructure.serialization import read_toml

app = typer.Typer(
    name="fab-agent",
    help="Create deterministic, review-only packages from straight pipe-spool sketches.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


async def _await_and_close[ResultT](
    dependencies: Dependencies,
    awaitable: Awaitable[ResultT],
    *,
    heartbeat: bool = False,
) -> ResultT:
    task = asyncio.ensure_future(awaitable)
    try:
        while heartbeat:
            done, _ = await asyncio.wait({task}, timeout=15)
            if done:
                break
            typer.echo("Still processing with Ollama Cloud...", err=True)
        return await task
    except BaseException:
        task.cancel()
        raise
    finally:
        await dependencies.model.aclose()


def _dependencies(config_path: Path | None) -> Dependencies:
    config = load_config(config_path)
    clock = SystemClock()
    return Dependencies(
        config=config,
        model=build_model_client(config),
        store=FilesystemRunStore(
            config.output.root,
            clock=clock,
            id_generator=SecureIdGenerator(),
        ),
        catalogs=load_catalogs(config.catalogs.root),
        clock=clock,
    )


def _fail(exc: Exception) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2)


def _result_payload(result: FabResult) -> dict[str, Any]:
    return result.model_dump(mode="json", exclude_none=True)


def _artifact_label(path: Path) -> str:
    if path.name == "review.md":
        return "Review"
    if path.name == "bom.csv":
        return "BOM"
    return {
        ".step": "Fusion CAD (STEP)",
        ".png": "Diagram",
        ".svg": "Diagram (SVG)",
    }.get(path.suffix.casefold(), "Artifact")


def _print_artifacts(paths: list[Path]) -> None:
    if not paths:
        return
    typer.echo("Artifacts:")
    for path in sorted(paths):
        typer.echo(f"  {_artifact_label(path):18} {path}")


def _print_result(result: FabResult, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(_result_payload(result), sort_keys=True))
        return
    typer.echo(f"Run:    {result.run_id}")
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Path:   {result.run_path}")
    if result.warnings:
        typer.echo("Warnings:")
        for warning in result.warnings:
            typer.echo(f"  - {warning}")
    if result.question:
        typer.echo(f"Question: {result.question}")
        typer.echo(f"Next: fab-agent resume {result.run_id}")
    elif result.status == "needs_review":
        typer.echo(f"Next: inspect {result.run_path / 'current.toml'}")
    elif result.status == "complete":
        typer.echo("REVIEW OUTPUT — NOT APPROVED FOR FABRICATION")
        _print_artifacts(list(result.artifacts.values()))


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", help="Configuration TOML path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Check configuration, catalogs, output, Ollama, and CadQuery."""

    try:
        dependencies = _dependencies(config)
        checks: dict[str, Any] = {
            "configuration": "ok",
            "catalogs": "ok",
            "output": "pending",
            "cadquery": "pending",
            "ollama": "pending",
        }
        dependencies.config.output.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dependencies.config.output.root):
            checks["output"] = "ok"
        import cadquery  # noqa: F401

        checks["cadquery"] = "ok"
        ollama_ready = True
        try:
            health = asyncio.run(_await_and_close(dependencies, dependencies.model.health()))
            checks["ollama"] = health
            ollama_ready = bool(health.get("vision_model_available"))
        except FabAgentError as exc:
            checks["ollama"] = {"status": "error", "detail": str(exc)}
            ollama_ready = False
        if json_output:
            typer.echo(json.dumps(checks, sort_keys=True))
        else:
            for name, value in checks.items():
                typer.echo(f"{name:14} {value if isinstance(value, str) else json.dumps(value)}")
        if not ollama_ready:
            raise typer.Exit(code=1)
    except (FabAgentError, OSError, ImportError) as exc:
        _fail(exc)


@app.command("run")
def run_command(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    design_type: str = typer.Option(
        "pipe_spool", "--design-type", help="Registered deterministic design handler."
    ),
    config: Path | None = typer.Option(None, "--config", help="Configuration TOML path."),
    demo: bool = typer.Option(False, "--demo", help="Permit synthetic demo catalog geometry."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Start a new run from an image."""

    try:
        dependencies = _dependencies(config)
        location = (
            "Ollama Cloud" if dependencies.config.ollama.connection == "cloud" else "local Ollama"
        )
        typer.echo(
            f"Processing with {location}; image reading may take a few minutes...",
            err=True,
        )
        result = asyncio.run(
            _await_and_close(
                dependencies,
                run_fab_agent(
                    FabRequest(input_image=image, design_type=design_type, demo=demo), dependencies
                ),
                heartbeat=dependencies.config.ollama.connection == "cloud",
            )
        )
        _print_result(result, json_output)
    except (FabAgentError, OSError) as exc:
        _fail(exc)


@app.command()
def resume(
    run_id: str = typer.Argument(...),
    answer: str | None = typer.Option(None, "--answer", help="Focused clarification answer."),
    answer_file: Path | None = typer.Option(
        None, "--answer-file", exists=True, dir_okay=False, readable=True
    ),
    config: Path | None = typer.Option(None, "--config", help="Configuration TOML path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Resume one run that is awaiting a focused answer."""

    if answer is not None and answer_file is not None:
        _fail(ValueError("Use either --answer or --answer-file, not both"))
    human_answer = answer or (
        answer_file.read_text(encoding="utf-8").strip() if answer_file else None
    )
    if human_answer is None:
        human_answer = typer.prompt("Answer")
    try:
        dependencies = _dependencies(config)
        typer.echo("Resuming deterministic workflow...", err=True)
        result = asyncio.run(
            _await_and_close(
                dependencies,
                resume_fab_agent(run_id, human_answer, dependencies),
            )
        )
        _print_result(result, json_output)
    except (FabAgentError, OSError) as exc:
        _fail(exc)


@app.command()
def show(
    run_id: str = typer.Argument(...),
    config: Path | None = typer.Option(None, "--config", help="Configuration TOML path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show current run status, validation, and artifact paths."""

    try:
        dependencies = _dependencies(config)
        current = dependencies.store.current(run_id)
        payload: dict[str, Any] = {"run_id": run_id, **current}
        artifact_paths: list[Path] = []
        if int(current["version"]) > 0:
            version_path = dependencies.store.active_version_path(run_id)
            payload["run_path"] = str(version_path.parent.parent)
            payload["validation"] = read_toml(version_path / "validation.toml")
            artifacts = version_path / "artifacts"
            artifact_paths = (
                [item for item in sorted(artifacts.rglob("*")) if item.is_file()]
                if artifacts.exists()
                else []
            )
            payload["artifacts"] = [str(path) for path in artifact_paths]
        if json_output:
            typer.echo(json.dumps(payload, sort_keys=True, default=str))
        else:
            typer.echo(f"Run:     {run_id}")
            typer.echo(f"Status:  {current['status']}")
            typer.echo(f"Version: {int(current['version']):03d}")
            if current.get("question"):
                typer.echo(f"Question: {current['question']}")
            for diagnostic in current.get("diagnostics", []):
                typer.echo(f"Diagnostic: {diagnostic}")
            _print_artifacts(artifact_paths)
        asyncio.run(dependencies.model.aclose())
    except (FabAgentError, OSError) as exc:
        _fail(exc)


@app.command()
def rebuild(
    run_id: str = typer.Argument(...),
    config: Path | None = typer.Option(None, "--config", help="Configuration TOML path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Create a new version from the editable draft without calling a model."""

    try:
        dependencies = _dependencies(config)
        result = asyncio.run(_await_and_close(dependencies, rebuild_run(run_id, dependencies)))
        _print_result(result, json_output)
    except (FabAgentError, OSError, ValueError) as exc:
        _fail(exc)
