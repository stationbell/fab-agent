"""Stable trigger-neutral public API."""

from __future__ import annotations

from fab_agent.application.runner import Dependencies, resume_run, start_run
from fab_agent.domain.results import FabRequest, FabResult


async def run_fab_agent(request: FabRequest, dependencies: Dependencies) -> FabResult:
    return await start_run(request, dependencies)


async def resume_fab_agent(run_id: str, human_answer: str, dependencies: Dependencies) -> FabResult:
    return await resume_run(run_id, human_answer, dependencies)
