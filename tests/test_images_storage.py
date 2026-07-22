from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from fab_agent.config import ImagesConfig
from fab_agent.domain.design import FabricationDesign
from fab_agent.domain.provenance import ProvenanceDocument
from fab_agent.errors import ImageInputError, StorageError
from fab_agent.infrastructure.filesystem import FilesystemRunStore, sanitize_slug
from fab_agent.infrastructure.images import normalize_image
from fab_agent.infrastructure.serialization import write_toml

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "sample-sketch-1.jpg",
        "sample-sketch-2.png",
        "sample-sketch-3.jpg",
        "sample-sketch-4.jpg",
    ],
)
def test_normalizes_real_images(name: str) -> None:
    normalized = normalize_image(ROOT / "fixtures" / name, ImagesConfig())
    assert max(normalized.width, normalized.height) <= 2000
    assert normalized.normalized_jpeg.startswith(b"\xff\xd8")


def test_rejects_filename_spoof(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_text("not an image")
    config = ImagesConfig(minimum_bytes=1)
    with pytest.raises(ImageInputError, match="not a supported image"):
        normalize_image(path, config)


def test_normalization_strips_exif(tmp_path: Path) -> None:
    source = tmp_path / "with-metadata.jpg"
    image = Image.new("RGB", (200, 100), "white")
    exif = Image.Exif()
    exif[0x010E] = "sensitive description"
    image.save(source, exif=exif)
    normalized = normalize_image(source, ImagesConfig(minimum_bytes=1))
    output = tmp_path / "normalized.jpg"
    output.write_bytes(normalized.normalized_jpeg)
    with Image.open(output) as reopened:
        assert len(reopened.getexif()) == 0


def test_filesystem_run_and_draft_round_trip(input_image: Path, tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path / "runs")
    run_id, run_path = store.create_run(
        source_image=input_image,
        source_extension=".png",
        normalized_jpeg=b"jpeg bytes",
        source_type="test",
        source_reference=None,
        metadata={},
        demo_mode=False,
    )
    store.save_draft(run_id, FabricationDesign(), ProvenanceDocument())
    design, provenance = store.load_draft(run_id)
    assert design == FabricationDesign()
    assert provenance == ProvenanceDocument()
    assert (run_path / "input" / "normalized.jpg").read_bytes() == b"jpeg bytes"
    assert len((run_path / "events.jsonl").read_text().splitlines()) == 1


def test_load_draft_accepts_and_discards_removed_provenance_metadata(
    input_image: Path,
    tmp_path: Path,
) -> None:
    store = FilesystemRunStore(tmp_path / "runs")
    run_id, run_path = store.create_run(
        source_image=input_image,
        source_extension=".png",
        normalized_jpeg=b"jpeg bytes",
        source_type="test",
        source_reference=None,
        metadata={},
        demo_mode=False,
    )
    store.save_draft(run_id, FabricationDesign(), ProvenanceDocument())
    write_toml(
        run_path / "draft" / "provenance.toml",
        {
            "schema_version": 1,
            "unresolved_low_confidence_fields": ["spools.S1.material_raw"],
            "entries": [
                {
                    "field_path": "spools.S1.material_raw",
                    "source_type": "image",
                    "raw_text": "pipe",
                    "recorded_at": datetime(2026, 7, 22, tzinfo=UTC).isoformat(),
                    "agent_step": 1,
                    "confidence": 0.5,
                    "region": {"x": 1, "y": 2, "width": 3, "height": 4},
                }
            ],
        },
    )

    _, provenance = store.load_draft(run_id)

    assert len(provenance.entries) == 1
    assert provenance.entries[0].field_path == "spools.S1.material_raw"
    assert "confidence" not in provenance.entries[0].model_dump()


def test_run_id_rejects_path_traversal(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path / "runs")
    with pytest.raises(StorageError, match="Invalid run ID"):
        store.current("../outside")


def test_slug_is_safe_and_bounded() -> None:
    assert sanitize_slug("../../My sketch !!") == "my-sketch"
    assert len(sanitize_slug("a" * 100)) == 48


def test_run_id_uses_injected_clock_and_id(input_image: Path, tmp_path: Path) -> None:
    class FixedClock:
        def now(self) -> datetime:
            return datetime(2026, 7, 21, 22, 54, 31, tzinfo=UTC)

    class FixedId:
        def new_id(self) -> str:
            return "fixed123"

    store = FilesystemRunStore(tmp_path / "runs", clock=FixedClock(), id_generator=FixedId())
    run_id, _ = store.create_run(
        source_image=input_image,
        source_extension=".png",
        normalized_jpeg=b"jpeg",
        source_type="test",
        source_reference=None,
        metadata={"origin": "fixture"},
        demo_mode=False,
    )
    assert run_id == "20260721T225431Z_fixed123_sketch"
