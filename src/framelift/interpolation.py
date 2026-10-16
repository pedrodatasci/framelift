"""In-between frames with RIFE, so a higher frame rate actually moves more smoothly.

``--output-fps`` alone repeats frames: the duration stays the same, but there are no new
images. With ``--interpolate``, the extra frames are drawn by RIFE (Real-Time Intermediate
Flow Estimation): it estimates how everything moves between two frames and renders the
image at any moment in between.

* :class:`FrameInterpolator` runs the network, through PyTorch (NVIDIA GPU or CPU) or
  OpenVINO (Intel GPU, or the CPU with ``--backend openvino``).
* :class:`Retimer` decides which output frames are originals and which are in-between
  ones, and at which moment, keeping exactly the frame count and duration that repeating
  frames gives. Across a scene cut it repeats instead, since there's no motion to follow.

It works on the finished frames (after the AI and the resize), so the AI still runs once
per original frame. The network is RIFE 4.26 from Practical-RIFE, the version its authors
recommend for most scenes.
"""

# RIFE's architecture below follows Practical-RIFE (https://github.com/hzwer/Practical-RIFE)
# by Zhewei Huang (hzwer) et al., MIT License. The weights are downloaded from the copy
# that vs-rife (https://github.com/HolyWu/vs-rife, MIT License) publishes on GitHub.

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .device import Device
from .filters import Frame
from .models import download_file, resolve_weights_dir
from .planning import SNAP

log = logging.getLogger(__name__)

RIFE_VERSION = "4.26"
RIFE_FILENAME = f"rife-v{RIFE_VERSION}.pkl"
RIFE_URL = f"https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{RIFE_VERSION}.pkl"

SCENE_CUT_SIMILARITY = 0.3
"""Below this similarity (SSIM of 32x32 grayscale thumbnails, 1 = identical) two frames
count as a scene cut and are repeated, not interpolated. Consecutive frames of a shot
score above 0.9; unrelated images, around 0.1."""


def ensure_rife_weights(weights_dir: str | Path | None = None) -> Path:
    """Return the local path to RIFE's weights, downloading them (25 MB) if needed."""
    path = resolve_weights_dir(weights_dir) / RIFE_FILENAME
    if not path.exists():
        download_file(RIFE_URL, path)
    return path


# -- the network -------------------------------------------------------------------


def _conv(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, 3, stride, 1), nn.LeakyReLU(0.2, True))


class _Head(nn.Module):
    """Small features of each frame, computed once and reused for every in-between frame."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn0(x.clamp(0.0, 1.0)))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class _ResConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class _IFBlock(nn.Module):
    """Refines the motion estimate at one scale."""

    def __init__(self, in_planes: int, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, channels // 2, 2), _conv(channels // 2, channels, 2)
        )
        self.convblock = nn.Sequential(*[_ResConv(channels) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(channels, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None, scale: float):
        x = functional.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        out = self.lastconv(self.convblock(self.conv0(x)))
        out = functional.interpolate(out, scale_factor=scale, mode="bilinear")
        return out[:, :4] * scale, out[:, 4:5], out[:, 5:]


class IFNet(nn.Module):
    """RIFE 4.26: estimates the motion between two frames, coarse to fine, and blends
    both frames moved to the requested moment."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.block0 = _IFBlock(7 + 8, 192)
        self.block1 = _IFBlock(8 + 4 + 8 + 8, 128)
        self.block2 = _IFBlock(8 + 4 + 8 + 8, 96)
        self.block3 = _IFBlock(8 + 4 + 8 + 8, 64)
        self.block4 = _IFBlock(8 + 4 + 8 + 8, 32)
        self.encode = _Head()
        self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

    def forward(self, img0, img1, timestep, f0, f1, grid, flow_div) -> torch.Tensor:
        img0, img1 = img0.clamp(0.0, 1.0), img1.clamp(0.0, 1.0)
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)
        flow = mask = feat = None
        warped0, warped1 = img0, img1
        for block, scale in zip(blocks, self.scale_list):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale)
            else:
                wf0 = _warp(f0, flow[:, :2], grid, flow_div)
                wf1 = _warp(f1, flow[:, 2:4], grid, flow_div)
                x = torch.cat((warped0, warped1, wf0, wf1, timestep, mask, feat), 1)
                delta, mask, feat = block(x, flow, scale)
                flow = flow + delta
            warped0 = _warp(img0, flow[:, :2], grid, flow_div)
            warped1 = _warp(img1, flow[:, 2:4], grid, flow_div)
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)


def _warp(image, flow, grid, flow_div) -> torch.Tensor:
    """Move ``image`` along ``flow`` (pixels). Always in FP32: FP16 isn't precise enough
    for sampling positions on large frames."""
    offsets = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1)
    positions = (grid + offsets.float()).permute(0, 2, 3, 1)
    moved = functional.grid_sample(
        image.float(), positions, mode="bilinear", padding_mode="border", align_corners=True
    )
    return moved.to(image.dtype)


class _Step(nn.Module):
    """One in-between frame for a fixed (padded) frame size; ``t`` has shape [1, 1, 1, 1]."""

    def __init__(self, network: IFNet, width: int, height: int) -> None:
        super().__init__()
        self.network = network
        horizontal = (
            torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(1, 1, height, width)
        )
        vertical = (
            torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(1, 1, height, width)
        )
        # Plain attributes, not buffers: they stay FP32 when the network goes FP16.
        self.grid = torch.cat([horizontal, vertical], 1)
        self.flow_div = torch.tensor([(width - 1.0) / 2.0, (height - 1.0) / 2.0])

    def forward(self, img0, img1, f0, f1, t) -> torch.Tensor:
        timestep = t.expand(-1, -1, img0.shape[2], img0.shape[3])
        return self.network(img0, img1, timestep, f0, f1, self.grid, self.flow_div)


def load_rife(weights_path: str | Path, scale: float = 1.0) -> IFNet:
    """Build RIFE and load its weights (training-only parts of the file are skipped)."""
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before 1.13
        state = torch.load(weights_path, map_location="cpu")
    network = IFNet(scale)
    wanted = network.state_dict()
    state = {k[len("module.") :]: v for k, v in state.items() if k.startswith("module.")}
    network.load_state_dict({k: v for k, v in state.items() if k in wanted}, strict=True)
    return network.eval()


# -- running it ------------------------------------------------------------------------


class FrameInterpolator:
    """Makes the frame at any moment between two frames of a given size.

    Each frame goes through :meth:`prepare` once (upload and features), then
    :meth:`between` draws as many in-between frames as needed from two prepared frames.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: Device,
        width: int,
        height: int,
        *,
        half: bool = False,
        backend: str = "torch",
    ) -> None:
        self.width, self.height = width, height
        # Above 1080p, RIFE estimates motion at half resolution (as its authors do for 4K).
        scale = 1.0 if width * height <= 1920 * 1088 else 0.5
        modulo = max(64, int(64 / scale))
        padded_w = math.ceil(width / modulo) * modulo
        padded_h = math.ceil(height / modulo) * modulo
        step = _Step(load_rife(weights_path, scale), padded_w, padded_h)

        use_openvino = device.kind == "igpu" or (backend == "openvino" and not device.is_cuda)
        if use_openvino:
            cache = Path(weights_path).parent / "openvino_cache"
            self._runner = _OpenVINORife(
                step, device, padded_w, padded_h, half=half, cache_dir=cache
            )
        else:
            self._runner = _TorchRife(step, device, padded_w, padded_h, half=half)

    def prepare(self, image: Frame) -> Any:
        """Get a BGR frame ready to be interpolated from."""
        return self._runner.prepare(image)

    def between(self, before: Any, after: Any, t: float) -> Frame:
        """The BGR frame at moment ``t`` (0 = ``before``, 1 = ``after``)."""
        return self._runner.between(before, after, t)[: self.height, : self.width]


class _TorchRife:
    def __init__(self, step: _Step, device: Device, width: int, height: int, *, half: bool) -> None:
        self._device = device.torch_device
        self._dtype = torch.float16 if half and device.is_cuda else torch.float32
        self._size = (width, height)
        step.network.to(self._device, self._dtype)
        step.grid, step.flow_div = step.grid.to(self._device), step.flow_div.to(self._device)
        self._step = step

    @torch.inference_mode()
    def prepare(self, image: Frame) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.ascontiguousarray(image)).to(self._device)
        x = x.permute(2, 0, 1).flip(0).unsqueeze(0).to(self._dtype).div_(255.0)  # BGR → RGB
        height, width = image.shape[:2]
        x = functional.pad(x, (0, self._size[0] - width, 0, self._size[1] - height))
        return x, self._step.network.encode(x)

    @torch.inference_mode()
    def between(self, before, after, t: float) -> Frame:
        timestep = torch.full((1, 1, 1, 1), float(t), dtype=self._dtype, device=self._device)
        out = self._step(before[0], after[0], before[1], after[1], timestep)
        out = out[0].float().clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
        return out.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


class _OpenVINORife:
    """RIFE through OpenVINO. The converted model is saved for each frame size, so later
    runs skip the conversion; compiled models go to OpenVINO's cache as for the AI."""

    def __init__(
        self, step: _Step, device: Device, width: int, height: int, *, half: bool, cache_dir: Path
    ) -> None:
        from . import device as devices  # read at call time (tests point it at the CPU)
        from .upscaler import import_openvino

        ov = import_openvino()
        self._size = (width, height)
        core = ov.Core()
        with contextlib.suppress(Exception):  # only a speed-up for the next runs
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})

        tag = f"rife-v{RIFE_VERSION}-{width}x{height}-s{step.network.scale_list[-1]:g}"
        image, feature = self._example(3), self._example(4)
        encode_model = self._model(ov, core, cache_dir / f"{tag}-encode.xml",
                                   step.network.encode, (image,))  # fmt: skip
        step_inputs = (image, image, feature, feature, torch.zeros(1, 1, 1, 1))
        step_model = self._model(ov, core, cache_dir / f"{tag}.xml", step, step_inputs)

        target = devices.OPENVINO_DEVICE if device.kind == "igpu" else "CPU"
        try:
            self._encode = self._compile(core, encode_model, target, half)
            self._step = self._compile(core, step_model, target, half)
        except Exception as exc:  # e.g. an old GPU driver without an operation RIFE needs
            if target == "CPU":
                raise
            log.warning("RIFE couldn't run on the Intel GPU (%s); using the CPU for it.", exc)
            self._encode = self._compile(core, encode_model, "CPU", half)
            self._step = self._compile(core, step_model, "CPU", half)

    def _example(self, channels: int) -> torch.Tensor:
        return torch.zeros(1, channels, self._size[1], self._size[0])

    @staticmethod
    def _model(ov, core, path: Path, module: nn.Module, example: tuple):
        if path.exists():
            with contextlib.suppress(Exception):
                return core.read_model(str(path))
        with torch.no_grad():
            model = ov.convert_model(module, example_input=example)
        with contextlib.suppress(Exception):  # caching only
            ov.save_model(model, str(path), compress_to_fp16=False)
        return model

    @staticmethod
    def _compile(core, model, target: str, half: bool):
        config = {
            "INFERENCE_PRECISION_HINT": "f16" if half and target != "CPU" else "f32",
            "PERFORMANCE_HINT": "LATENCY",
        }
        if target == "CPU":
            config["INFERENCE_NUM_THREADS"] = torch.get_num_threads()
        return core.compile_model(model, target, config).create_infer_request()

    def prepare(self, image: Frame) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        x = np.zeros((1, 3, self._size[1], self._size[0]), np.float32)
        x[0, :, :height, :width] = image[..., ::-1].transpose(2, 0, 1)  # BGR → RGB, CHW
        x /= 255.0
        self._encode.infer([x])
        return x, self._encode.get_output_tensor(0).data.copy()

    def between(self, before, after, t: float) -> Frame:
        timestep = np.full((1, 1, 1, 1), t, np.float32)
        self._step.infer([before[0], after[0], before[1], after[1], timestep])
        out = self._step.get_output_tensor(0).data[0]
        out = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(out[::-1].transpose(1, 2, 0))  # RGB → BGR, HWC


# -- placing the in-between frames -------------------------------------------------------


@dataclass
class _Held:
    image: Frame
    source: Frame
    prepared: Any
    thumbnail: np.ndarray


class Retimer:
    """Turns the finished frames into the output frames at the new frame rate.

    The output has exactly as many frames as repeating would give (so the duration and
    the audio sync don't change); output frame ``k`` shows the moment ``k / ratio``, in
    input frames. Moments on (or within :data:`SNAP` of) an original frame show it;
    the others are drawn by RIFE, except across a scene cut or between two identical
    frames, where the nearest original is repeated.
    """

    def __init__(self, input_fps: float, output_fps: float, interpolator: FrameInterpolator):
        self._ratio = output_fps / input_fps
        self._interpolator = interpolator
        self._pushed = 0
        self._next = 0  # next output frame, counted from the start of this run
        self._current: _Held | None = None
        self.interpolated = 0
        """In-between frames drawn by RIFE so far."""
        self.scene_cuts = 0
        """Scene cuts found so far (their in-between frames are repeats)."""
        self.unchanged = 0
        """In-between frames that are repeats because both frames were identical."""

    def push(self, image: Frame, source: Frame) -> list[Frame]:
        """Take the next finished frame (``source`` is the same frame before the AI, for
        spotting scene cuts cheaply) and return the output frames that are now ready."""
        index = self._pushed
        self._pushed += 1
        previous = self._current
        # Nothing moves between identical frames (a still, or a frame the AI reused):
        # no need to prepare this one, nor to draw anything in between.
        same = previous is not None and (
            image is previous.image or np.array_equal(source, previous.source)
        )
        if same:
            current = _Held(image, source, previous.prepared, previous.thumbnail)
        else:
            prepared = self._interpolator.prepare(image)
            current = _Held(image, source, prepared, _thumbnail(source))
        self._current = current

        cut = (
            not same
            and previous is not None
            and (similarity(previous.thumbnail, current.thumbnail) < SCENE_CUT_SIMILARITY)
        )
        self.scene_cuts += cut

        ready = []
        # Every output moment up to this frame's own can be made now; later ones wait
        # for the next frame. ``t`` is the moment between the previous frame and this one.
        while self._next <= index * self._ratio + 1e-9:
            t = self._next / self._ratio - (index - 1)
            if previous is None or t >= 1 - SNAP:
                ready.append(current.image)
            elif t <= SNAP:
                ready.append(previous.image)
            elif cut or same:
                # The nearest original; at exactly halfway, the earlier one (as repeating does).
                ready.append(previous.image if t <= 0.5 else current.image)
                self.unchanged += same
            else:
                ready.append(self._interpolator.between(previous.prepared, current.prepared, t))
                self.interpolated += 1
            self._next += 1
        return ready

    def flush(self) -> tuple[Frame | None, int]:
        """The last frame and how many more times to write it: the rest of its share of
        the output, as there's no next frame to move towards."""
        total = round(self._pushed * self._ratio)
        copies, self._next = total - self._next, total
        return (self._current.image if self._current else None), copies


def _thumbnail(image: Frame) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM of two small grayscale images: 1 = identical, around 0 = unrelated."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mean_a, mean_b = blur(a), blur(b)
    var_a = blur(a * a) - mean_a * mean_a
    var_b = blur(b * b) - mean_b * mean_b
    covariance = blur(a * b) - mean_a * mean_b
    ssim = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2)) / (
        (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
    )
    return float(ssim.mean())
