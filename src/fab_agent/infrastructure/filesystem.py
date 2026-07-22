"""Filesystem-backed run storage with immutable version snapshots."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock, Timeout

from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.provenance import ProvenanceDocument
from fab_agent.domain.takeoff import Takeoff
from fab_agent.domain.validation import ValidationReport
from fab_agent.errors import RunStateError, StorageError
from fab_agent.infrastructure.serialization import (
    atomic_write_bytes,
    atomic_write_text,
    read_toml,
    stable_json,
    write_toml,
)
from fab_agent.ports import Clock, IdGenerator

RunStatus = Literal["processing", "complete", "awaiting_input", "needs_review"]


def utc_now() -> datetime:
    return datetime.now(UTC)


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:48] or "image"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            identifier = item.get("id") if isinstance(item, dict) else None
            key = str(identifier) if identifier else str(index)
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(item, path))
    else:
        flattened[prefix] = value
    return flattened


class FilesystemRunStore:
    def __init__(
        self,
        root: Path,
        *,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
    ) -> None:
        self.root = root.resolve()
        self._clock = clock
        self._id_generator = id_generator

    def _now(self) -> datetime:
        return self._clock.now() if self._clock is not None else utc_now()

    def _new_id(self) -> str:
        return (
            self._id_generator.new_id() if self._id_generator is not None else secrets.token_hex(5)
        )

    def _run_path(self, run_id: str) -> Path:
        if Path(run_id).name != run_id or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise StorageError("Invalid run ID")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise StorageError("Run path escapes output root")
        return path

    def _lock(self, run_id: str) -> FileLock:
        return FileLock(self._run_path(run_id) / ".run.lock", timeout=5)

    def create_run(
        self,
        *,
        source_image: Path,
        source_extension: str,
        normalized_jpeg: bytes,
        source_type: str,
        source_reference: str | None,
        metadata: dict[str, str],
        demo_mode: bool,
    ) -> tuple[str, Path]:
        now = self._now()
        slug = sanitize_slug(source_image.stem)
        for _ in range(10):
            unique = self._new_id()
            run_id = f"{now:%Y%m%dT%H%M%SZ}_{unique}_{slug}"
            run_path = self.root / run_id
            try:
                run_path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            input_path = run_path / "input"
            draft_path = run_path / "draft"
            (run_path / "versions").mkdir()
            input_path.mkdir()
            draft_path.mkdir()
            source_bytes = source_image.read_bytes()
            safe_extension = (
                source_extension
                if source_extension in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"}
                else ".img"
            )
            atomic_write_bytes(input_path / f"original{safe_extension}", source_bytes)
            atomic_write_bytes(input_path / "normalized.jpg", normalized_jpeg)
            write_toml(
                run_path / "run.toml",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "created_at": now.isoformat(),
                    "source_type": source_type,
                    "source_reference": source_reference or "",
                    "source_filename": source_image.name,
                    "demo_mode": demo_mode,
                    "metadata": metadata,
                },
            )
            self._write_current(run_path, status="processing", version=0)
            atomic_write_text(run_path / "events.jsonl", "")
            self.append_event(run_id, "run.created", {"source_type": source_type})
            return run_id, run_path
        raise StorageError("Could not generate a collision-resistant run ID")

    def _write_current(
        self,
        run_path: Path,
        *,
        status: RunStatus,
        version: int,
        question: str | None = None,
        target_field: str | None = None,
        questions_asked: int = 0,
        diagnostics: list[str] | None = None,
    ) -> None:
        write_toml(
            run_path / "current.toml",
            {
                "schema_version": 1,
                "status": status,
                "version": version,
                "question": question or "",
                "target_field": target_field or "",
                "questions_asked": questions_asked,
                "diagnostics": diagnostics or [],
                "updated_at": self._now().isoformat(),
            },
        )

    def current(self, run_id: str) -> dict[str, Any]:
        return read_toml(self._run_path(run_id) / "current.toml")

    def run_metadata(self, run_id: str) -> dict[str, Any]:
        return read_toml(self._run_path(run_id) / "run.toml")

    def save_draft(
        self, run_id: str, design: FabricationDesign, provenance: ProvenanceDocument
    ) -> None:
        run_path = self._run_path(run_id)
        try:
            with self._lock(run_id):
                write_toml(
                    run_path / "draft" / "design.toml",
                    design.model_dump(mode="json", exclude_none=True),
                )
                write_toml(
                    run_path / "draft" / "provenance.toml",
                    provenance.model_dump(mode="json", exclude_none=True),
                )
        except Timeout as exc:
            raise StorageError(f"Run {run_id} is busy") from exc

    def load_draft(self, run_id: str) -> tuple[FabricationDesign, ProvenanceDocument]:
        draft = self._run_path(run_id) / "draft"
        provenance_data = read_toml(draft / "provenance.toml")
        # Runs created before the flat transcription cleanup may contain unused
        # model-confidence and crop metadata. Accept them once, then save the
        # smaller current schema on the next write.
        provenance_data.pop("unresolved_low_confidence_fields", None)
        for entry in provenance_data.get("entries", []):
            if isinstance(entry, dict):
                entry.pop("confidence", None)
                entry.pop("region", None)
                entry.pop("agent_step", None)
        return (
            FabricationDesign.model_validate(read_toml(draft / "design.toml")),
            ProvenanceDocument.model_validate(provenance_data),
        )

    def normalized_image_path(self, run_id: str) -> Path:
        return self._run_path(run_id) / "input" / "normalized.jpg"

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        run_path = self._run_path(run_id)
        event = {
            "recorded_at": self._now().isoformat(),
            "type": event_type,
            "payload": payload,
        }
        try:
            with (run_path / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(stable_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise StorageError(f"Cannot append event for {run_id}: {exc}") from exc

    def set_status(self, run_id: str, status: RunStatus) -> None:
        run_path = self._run_path(run_id)
        try:
            with self._lock(run_id):
                current = self.current(run_id)
                self._write_current(
                    run_path,
                    status=status,
                    version=int(current["version"]),
                    questions_asked=int(current["questions_asked"]),
                )
        except Timeout as exc:
            raise StorageError(f"Run {run_id} is busy") from exc

    def commit_version(
        self,
        run_id: str,
        *,
        status: Literal["complete", "awaiting_input", "needs_review"],
        design: FabricationDesign,
        provenance: ProvenanceDocument,
        validation: ValidationReport,
        takeoff: Takeoff | None,
        artifacts_source: Path | None,
        question: str | None = None,
        target_field: str | None = None,
        diagnostics: list[str] | None = None,
    ) -> tuple[int, Path]:
        run_path = self._run_path(run_id)
        try:
            with self._lock(run_id):
                current = self.current(run_id)
                version = int(current["version"]) + 1
                version_name = f"{version:03d}"
                staging = run_path / "versions" / f".{version_name}.staging"
                final = run_path / "versions" / version_name
                if staging.exists() or final.exists():
                    raise StorageError(f"Version {version_name} already exists")
                staging.mkdir()
                write_toml(
                    staging / "design.toml", design.model_dump(mode="json", exclude_none=True)
                )
                write_toml(
                    staging / "provenance.toml",
                    provenance.model_dump(mode="json", exclude_none=True),
                )
                write_toml(
                    staging / "validation.toml",
                    validation.model_dump(mode="json", exclude_none=True),
                )
                if takeoff is not None:
                    write_toml(
                        staging / "takeoff.toml", takeoff.model_dump(mode="json", exclude_none=True)
                    )
                write_toml(
                    staging / "status.toml",
                    {
                        "status": status,
                        "created_at": self._now().isoformat(),
                        "parent_version": int(current["version"]),
                        "question": question or "",
                        "target_field": target_field or "",
                        "diagnostics": diagnostics or [],
                    },
                )
                write_toml(
                    staging / "derived.toml",
                    {
                        "schema_version": 1,
                        "input_version": version,
                        "functions": [
                            "fab_agent.domain.validation.validate_design",
                            *(
                                ["fab_agent.domain.takeoff.compute_takeoff"]
                                if takeoff is not None
                                else []
                            ),
                            *(
                                ["fab_agent.infrastructure.artifacts.generate_artifacts"]
                                if artifacts_source is not None
                                else []
                            ),
                        ],
                    },
                )
                self._write_changes(run_path, staging, design, int(current["version"]))
                if artifacts_source and artifacts_source.exists():
                    shutil.copytree(artifacts_source, staging / "artifacts")
                self._write_manifest(staging)
                os.replace(staging, final)
                questions_asked = int(current["questions_asked"])
                if status == "awaiting_input":
                    questions_asked += 1
                self._write_current(
                    run_path,
                    status=status,
                    version=version,
                    question=question,
                    target_field=target_field,
                    questions_asked=questions_asked,
                    diagnostics=diagnostics,
                )
                return version, final
        except Timeout as exc:
            raise StorageError(f"Run {run_id} is busy") from exc

    def _write_manifest(self, version_path: Path) -> None:
        files: list[dict[str, str]] = []
        for path in sorted(item for item in version_path.rglob("*") if item.is_file()):
            if path.name == "manifest.toml":
                continue
            files.append(
                {
                    "path": path.relative_to(version_path).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        write_toml(version_path / "manifest.toml", {"schema_version": 1, "files": files})

    def _write_changes(
        self,
        run_path: Path,
        staging: Path,
        design: FabricationDesign,
        parent_version: int,
    ) -> None:
        current_data = design.model_dump(mode="json", exclude_none=True)
        previous_data: dict[str, Any] = {}
        if parent_version > 0:
            previous_data = read_toml(
                run_path / "versions" / f"{parent_version:03d}" / "design.toml"
            )
        before = _flatten(previous_data)
        after = _flatten(current_data)
        changes = [
            {
                "field_path": path,
                "before": str(before.get(path, "<missing>")),
                "after": str(after.get(path, "<missing>")),
            }
            for path in sorted(before.keys() | after.keys())
            if before.get(path, "<missing>") != after.get(path, "<missing>")
        ]
        write_toml(
            staging / "changes.toml",
            {"schema_version": 1, "parent_version": parent_version, "changes": changes},
        )

    def require_awaiting_input(self, run_id: str) -> dict[str, Any]:
        current = self.current(run_id)
        if current["status"] != "awaiting_input":
            raise RunStateError(f"Run {run_id} is not awaiting input")
        return current

    def active_version_path(self, run_id: str) -> Path:
        current = self.current(run_id)
        version = int(current["version"])
        if version < 1:
            raise RunStateError(f"Run {run_id} has no committed version")
        return self._run_path(run_id) / "versions" / f"{version:03d}"

    def load_active_design(self, run_id: str) -> FabricationDesign:
        return FabricationDesign.model_validate(
            read_toml(self.active_version_path(run_id) / "design.toml")
        )
