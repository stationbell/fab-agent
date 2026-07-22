"""Deterministic image validation and normalization."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from fab_agent.config import ImagesConfig
from fab_agent.errors import ImageInputError

_FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "GIF": ".gif",
    "WEBP": ".webp",
    "HEIF": ".heic",
    "HEIC": ".heic",
}


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    source_format: str
    source_extension: str
    normalized_jpeg: bytes
    width: int
    height: int


def _register_heif_if_available() -> None:
    try:
        from pillow_heif import register_heif_opener  # type: ignore[import-not-found]
    except ImportError:
        return
    register_heif_opener()


def normalize_image(path: Path, config: ImagesConfig) -> NormalizedImage:
    if not path.is_file():
        raise ImageInputError(f"Input image does not exist or is not a file: {path}")
    size = path.stat().st_size
    if size < config.minimum_bytes:
        raise ImageInputError(f"Image is too small ({size} bytes; minimum {config.minimum_bytes})")
    if size > config.maximum_bytes:
        raise ImageInputError(f"Image is too large ({size} bytes; maximum {config.maximum_bytes})")

    _register_heif_if_available()
    try:
        with Image.open(path) as opened:
            source_format = (opened.format or "").upper()
            normalized_name = "jpeg" if source_format == "JPEG" else source_format.lower()
            if normalized_name in {"heif", "heic"} and "heic" not in config.allowed_formats:
                raise ImageInputError("HEIC images are disabled by configuration")
            if normalized_name not in config.allowed_formats:
                raise ImageInputError(
                    f"Detected image format {source_format or 'unknown'} is not allowed"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ImageInputError("Image dimensions are invalid")
            image.thumbnail(
                (config.max_edge_pixels, config.max_edge_pixels), Image.Resampling.LANCZOS
            )
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            return NormalizedImage(
                source_format=source_format,
                source_extension=_FORMAT_EXTENSIONS.get(source_format, ".img"),
                normalized_jpeg=output.getvalue(),
                width=image.width,
                height=image.height,
            )
    except ImageInputError:
        raise
    except UnidentifiedImageError as exc:
        if path.suffix.casefold() in {".heic", ".heif"}:
            raise ImageInputError(
                "HEIC decoder unavailable; install with `uv sync --extra heic`"
            ) from exc
        raise ImageInputError("File content is not a supported image") from exc
    except (OSError, ValueError) as exc:
        raise ImageInputError(f"Cannot decode image safely: {exc}") from exc
