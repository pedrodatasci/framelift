"""The Real-ESRGAN models framelift knows about, and how to fetch their weights.

This module is intentionally free of PyTorch imports: it only describes
models and downloads files, so it's cheap to import (e.g. for ``--help``).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm

log = logging.getLogger(__name__)

_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to download and build one Real-ESRGAN network."""

    name: str
    filename: str
    url: str
    architecture: Literal["rrdb", "srvgg"]
    native_scale: int
    """The upscale factor the network was trained for. framelift resizes to your
    requested ``scale`` afterwards, so this is not the final output size."""

    depth: int
    """``num_block`` for RRDB networks, ``num_conv`` for SRVGG networks."""

    summary: str


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="realesrgan-x4plus",
            filename="RealESRGAN_x4plus.pth",
            url=f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
            architecture="rrdb",
            native_scale=4,
            depth=23,
            summary="Best detail on real-world footage. Heavy and slow.",
        ),
        ModelSpec(
            name="realesrnet-x4plus",
            filename="RealESRNet_x4plus.pth",
            url=f"{_RELEASES}/v0.1.1/RealESRNet_x4plus.pth",
            architecture="rrdb",
            native_scale=4,
            depth=23,
            summary="Like x4plus but smoother, with fewer invented textures.",
        ),
        ModelSpec(
            name="realesrgan-x4plus-anime-6B",
            filename="RealESRGAN_x4plus_anime_6B.pth",
            url=f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            architecture="rrdb",
            native_scale=4,
            depth=6,
            summary="Tuned for anime and illustrations. Lighter than x4plus.",
        ),
        ModelSpec(
            name="realesr-animevideov3",
            filename="realesr-animevideov3.pth",
            url=f"{_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
            architecture="srvgg",
            native_scale=4,
            depth=16,
            summary="Tiny and fast, made for video. Great first choice.",
        ),
    )
}


def get_model(name: str) -> ModelSpec:
    """Look up a model by name, with a helpful error for typos."""
    try:
        return MODELS[name]
    except KeyError:
        choices = ", ".join(MODELS)
        raise ValueError(f"Unknown model '{name}'. Available: {choices}.") from None


def ensure_weights(model: ModelSpec, weights_dir: str | Path) -> Path:
    """Return the local path to ``model``'s weights, downloading them if needed."""
    weights_path = Path(weights_dir) / model.filename

    if not weights_path.exists():
        download_file(model.url, weights_path)

    return weights_path


def download_file(url: str, destination: str | Path) -> Path:
    """Download ``url`` to ``destination`` with a progress bar.

    The file is first written to ``<name>.part`` and only renamed once the
    download completes, so an interrupted download never leaves behind a
    truncated file that would be mistaken for valid weights next time.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    log.info("Downloading %s (first run only)…", destination.name)
    log.debug("  from %s", url)
    log.debug("  to   %s", destination)

    with _DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=destination.name) as bar:
        try:
            urllib.request.urlretrieve(url, filename=partial, reporthook=bar.update_to)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    partial.replace(destination)
    return destination


class _DownloadProgress(tqdm):
    """Adapts tqdm to ``urlretrieve``'s ``reporthook(block, block_size, total)``."""

    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int = None) -> None:
        if total_size is not None and total_size > 0:
            self.total = total_size
        self.update(block_num * block_size - self.n)
