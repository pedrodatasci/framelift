"""A thin, friendly wrapper around Real-ESRGAN's ``RealESRGANer``."""

from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path

import torch

from .device import Device
from .errors import MissingDependencyError, OutOfMemoryError
from .filters import Frame
from .models import ModelSpec


class _Discard(io.TextIOBase):
    def write(self, text: str) -> int:
        return len(text)


_SILENT = _Discard()

INSTALL_HINT = "pip install realesrgan basicsr facexlib gfpgan opencv-python tqdm pillow"


def _patch_torchvision_for_basicsr() -> None:
    """Keep ``basicsr`` importable on modern torchvision.

    ``basicsr`` imports ``torchvision.transforms.functional_tensor``, a module
    torchvision removed in 0.17. We register a tiny stand-in that provides the
    one function basicsr needs — but only if the real module is really gone.
    """
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return  # older torchvision: the real module still exists
    except ImportError:
        pass

    try:
        import torchvision.transforms.functional as functional
    except ImportError:
        return  # torchvision missing entirely; the import below will explain

    shim = types.ModuleType(module_name)
    shim.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module_name] = shim


def _import_realesrgan():
    _patch_torchvision_for_basicsr()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan import RealESRGANer
    except Exception as exc:  # basicsr can fail with more than ImportError
        raise MissingDependencyError(
            f"Couldn't import the Real-ESRGAN packages ({exc}).\n"
            f"Install them with:\n  {INSTALL_HINT}"
        ) from exc

    return RRDBNet, SRVGGNetCompact, RealESRGANer


def build_network(spec: ModelSpec) -> torch.nn.Module:
    """Create the (untrained) network architecture that matches ``spec``."""
    RRDBNet, SRVGGNetCompact, _ = _import_realesrgan()

    if spec.architecture == "rrdb":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=spec.depth,
            num_grow_ch=32,
            scale=spec.native_scale,
        )

    if spec.architecture == "srvgg":
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=spec.depth,
            upscale=spec.native_scale,
            act_type="prelu",
        )

    raise ValueError(f"Unsupported architecture: {spec.architecture}")


class AIUpscaler:
    """Loads a Real-ESRGAN model once and upscales frames with it."""

    def __init__(
        self,
        spec: ModelSpec,
        weights_path: Path,
        device: Device,
        *,
        tile: int = 256,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = False,
    ) -> None:
        _, _, RealESRGANer = _import_realesrgan()

        self.spec = spec
        self.device = device
        self.half = bool(half and device.is_cuda)  # FP16 only makes sense on GPU

        self._engine = RealESRGANer(
            scale=spec.native_scale,
            model_path=str(weights_path),
            model=build_network(spec),
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=self.half,
            # Passed explicitly: otherwise RealESRGANer grabs CUDA whenever it
            # exists, even if the user asked for --device cpu.
            device=device.torch_device,
            gpu_id=device.gpu_id,
        )

    def upscale(self, frame: Frame, scale: float) -> Frame:
        """Return ``frame`` enlarged by ``scale`` (relative to its current size)."""
        try:
            # RealESRGANer prints "Tile 1/N" for every tile, which garbles the
            # progress bar; we swallow that chatter.
            with torch.inference_mode(), contextlib.redirect_stdout(_SILENT):
                enhanced, _ = self._engine.enhance(frame, outscale=scale)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise OutOfMemoryError(
                    "Ran out of memory while upscaling. Try a smaller tile, e.g. --tile 128."
                ) from exc
            raise

        return enhanced
