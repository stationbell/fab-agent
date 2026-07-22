from pathlib import Path

import pytest
from typer.testing import CliRunner

from fab_agent.adapters.cli import _print_result, app
from fab_agent.domain.results import FabResult


def test_cli_help_is_clear() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "review-only packages" in result.stdout
    assert "doctor" in result.stdout
    assert "resume" in result.stdout


def test_run_help_exposes_design_handler_selection() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--design-type" in result.stdout
    assert "pipe_spool" in result.stdout


def test_complete_result_prints_cad_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cad = tmp_path / "S1.step"
    result = FabResult(
        run_id="run-1",
        status="complete",
        run_path=tmp_path,
        artifacts={"S1_step": cad},
    )

    _print_result(result, json_output=False)

    output = capsys.readouterr().out
    assert "CAD (STEP)" in output
    assert str(cad) in output


def test_result_prints_diagnostics_without_requiring_file_inspection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = FabResult(
        run_id="run-1",
        status="needs_review",
        run_path=tmp_path,
        warnings=["cloud credential is missing"],
    )

    _print_result(result, json_output=False)

    assert "cloud credential is missing" in capsys.readouterr().out
