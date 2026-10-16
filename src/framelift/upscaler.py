"""A thin, friendly wrapper around Real-ESRGAN's ``RealESRGANer``."""

from __future__ import annotations

import contextlib
import math
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional

from .archs import RRDBNet, SRVGGNetCompact
from .device import INSTALL_OPENVINO, Device
from .errors import MissingDependencyError, OutOfMemoryError
from .filters import Frame
from .models import ModelSpec


def build_network(spec: ModelSpec) -> torch.nn.Module:
    """Create the (untrained) network architecture that matches ``spec``."""

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


def load_network(spec: ModelSpec, weights_path: Path) -> torch.nn.Module:
    """``spec``'s network with its trained weights, ready for inference (on the CPU)."""
    network = build_network(spec)
    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 1.13 has no weights_only
        state = torch.load(str(weights_path), map_location="cpu")
    key = "params_ema" if "params_ema" in state else "params"
    network.load_state_dict(state[key], strict=True)
    return network.eval()


class AIUpscaler:
    """Loads a Real-ESRGAN model once and upscales frames with it.

    The work is split in two, so a run can overlap them: :meth:`infer` runs the AI and
    returns the frame at the model's own 4x size; :meth:`finish` resizes that to the
    requested scale. :meth:`upscale` does both. :meth:`infer` keeps no state between
    calls, so several threads can use it at once.

    It does what Real-ESRGAN's ``RealESRGANer.enhance`` does for a video frame (same
    tiling, padding and rounding, so the same output), minus its per-call state and
    console output, and converts the result to 8 bits on the device itself.
    """

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
        backend: str = "torch",
    ) -> None:
        self.spec = spec
        self.device = device
        self.scale = spec.native_scale
        self.tile, self.tile_pad, self.pre_pad = tile, tile_pad, pre_pad
        self.half = bool(half and device.is_gpu)  # FP16 only makes sense on a GPU
        # Real-ESRGAN pads frames so their size divides evenly, for 2x and 1x models.
        self.mod_scale = {2: 2, 1: 4}.get(self.scale)

        network = load_network(spec, Path(weights_path))
        self.channels_last = False
        self.backend = "openvino" if device.kind == "igpu" else backend
        if self.backend == "openvino" and not device.is_cuda:
            from . import device as devices  # read at call time (tests point it at the CPU)

            target = devices.OPENVINO_DEVICE if device.kind == "igpu" else "CPU"
            cache = Path(weights_path).parent / "openvino_cache"
            self._model = OpenVINOModel(network, target, half=self.half, cache_dir=cache)
            return

        network = network.to(device.torch_device)
        if self.half:
            network = network.half()
        # On CPU, the compact video network runs ~1.5x faster with its weights in
        # channels-last layout (oneDNN's native one). The heavier RRDB networks get
        # slightly *slower* that way, so they're left as they are.
        self.channels_last = not device.is_cuda and spec.architecture == "srvgg"
        if self.channels_last:
            network = network.to(memory_format=torch.channels_last)
        self._model = network

    def upscale(self, frame: Frame, scale: float) -> Frame:
        """Return ``frame`` enlarged by ``scale`` (relative to its current size)."""
        return self.finish(self.infer(frame), frame.shape, scale)

    def infer(self, frame: Frame) -> Frame:
        """Run the AI on a BGR frame; returns it at the model's native scale (4x)."""
        try:
            if isinstance(self._model, OpenVINOModel):
                return self._infer_8bit(frame)
            with torch.inference_mode():
                output = self._run(self._to_tensor(frame))
                # Padding was only added to the right and bottom: keep the frame itself.
                height, width = frame.shape[:2]
                output = output[:, :, : height * self.scale, : width * self.scale]
                return self._to_image(output)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "out_of_resources" in message:
                raise OutOfMemoryError(
                    "Ran out of memory while upscaling. Try a smaller tile, e.g. --tile 128."
                ) from exc
            raise

    def finish(self, native: Frame, source_shape, scale: float) -> Frame:
        """Resize :meth:`infer`'s output to ``scale`` times the source size."""
        height, width = source_shape[:2]
        if scale == float(self.scale):
            return native
        size = (int(width * scale), int(height * scale))
        return cv2.resize(native, size, interpolation=cv2.INTER_LANCZOS4)

    # -- the steps of RealESRGANer.enhance, for 8-bit BGR frames -------------------------

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return self._tiled(image) if self.tile > 0 else self._model(image)

    def _tiles(self, height: int, width: int):
        """Real-ESRGAN's tiling: for each tile, the padded input box ``(top, bottom, left,
        right)``, where its result goes in the output, and which part of the result to keep."""
        scale, tile, pad = self.scale, self.tile, self.tile_pad
        for y in range(math.ceil(height / tile)):
            for x in range(math.ceil(width / tile)):
                left, top = x * tile, y * tile
                right, bottom = min(left + tile, width), min(top + tile, height)
                left_pad, top_pad = max(left - pad, 0), max(top - pad, 0)
                right_pad, bottom_pad = min(right + pad, width), min(bottom + pad, height)
                inner_x, inner_y = (left - left_pad) * scale, (top - top_pad) * scale
                rows, cols = (bottom - top) * scale, (right - left) * scale
                yield (
                    (top_pad, bottom_pad, left_pad, right_pad),
                    (slice(top * scale, bottom * scale), slice(left * scale, right * scale)),
                    (slice(inner_y, inner_y + rows), slice(inner_x, inner_x + cols)),
                )

    def _infer_8bit(self, frame: Frame) -> Frame:
        """The OpenVINO path: the same padding and tiling as :meth:`_to_tensor` and
        :meth:`_tiled`, on the 8-bit frame itself. The model converts to and from floats."""
        height, width = frame.shape[:2]
        image = np.ascontiguousarray(frame)
        if self.pre_pad:
            image = np.pad(image, ((0, self.pre_pad), (0, self.pre_pad), (0, 0)), "reflect")
        if self.mod_scale:
            pad_h = (self.mod_scale - image.shape[0] % self.mod_scale) % self.mod_scale
            pad_w = (self.mod_scale - image.shape[1] % self.mod_scale) % self.mod_scale
            if pad_h or pad_w:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), "reflect")

        if self.tile > 0:
            rows, cols = image.shape[:2]
            output = np.empty((rows * self.scale, cols * self.scale, 3), np.uint8)
            for (top, bottom, left, right), target, keep in self._tiles(rows, cols):
                output[target] = self._model(image[top:bottom, left:right])[keep]
        else:
            output = self._model(image)

        if output.shape[:2] == (height * self.scale, width * self.scale):
            return output
        return np.ascontiguousarray(output[: height * self.scale, : width * self.scale])

    def _to_tensor(self, frame: Frame) -> torch.Tensor:
        # uint8 goes to the device (4x less to copy than float), then BGR → RGB in [0, 1].
        image = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device.torch_device)
        image = image.permute(2, 0, 1).flip(0).float().div_(255).unsqueeze(0)
        if self.half and self.device.is_cuda:
            image = image.half()
        if self.pre_pad:
            image = functional.pad(image, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        if self.mod_scale:
            height, width = image.shape[2:]
            pad_h = (self.mod_scale - height % self.mod_scale) % self.mod_scale
            pad_w = (self.mod_scale - width % self.mod_scale) % self.mod_scale
            if pad_h or pad_w:
                image = functional.pad(image, (0, pad_w, 0, pad_h), "reflect")
        return image

    def _tiled(self, image: torch.Tensor) -> torch.Tensor:
        """Upscale tile by tile; each tile sees ``tile_pad`` extra pixels around it."""
        _, channels, height, width = image.shape
        output = image.new_zeros((1, channels, height * self.scale, width * self.scale))
        for (top, bottom, left, right), (rows, cols), (keep_rows, keep_cols) in self._tiles(
            height, width
        ):
            result = self._model(image[:, :, top:bottom, left:right])
            output[:, :, rows, cols] = result[:, :, keep_rows, keep_cols]
        return output

    @staticmethod
    def _to_image(output: torch.Tensor) -> Frame:
        # Same rounding as Real-ESRGAN (float32, clamp, x255, round half to even), done
        # on the device, which also makes the copy back 4x smaller.
        image = output.squeeze(0).float().clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
        return image.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


class OpenVINOModel:
    """A Real-ESRGAN network compiled by OpenVINO, for an Intel GPU or the CPU.

    Called with an 8-bit BGR image ``(height, width, 3)``, as OpenCV holds it, and returns
    the 8-bit BGR result at the model's scale. The conversions around the network (to
    [0, 1] floats on the way in; Real-ESRGAN's clamp, x255 and round-half-to-even on the
    way out) are compiled into the model, so on a GPU they run there too, the CPU is left
    alone, and a quarter of the data comes back (8 bits per value instead of 32).

    The BGR ↔ RGB swaps are done here, on the 8-bit arrays, not in the model: OpenVINO
    folds a channel swap into the network's first or last layer, and with the x4plus
    networks (RRDBNet) that fold came out wrong, with red and blue mixed up.

    Each tile size gets compiled once (a few seconds on a GPU, cached on disk for later
    runs), and each thread gets its own inference request, so several workers can use it.
    """

    def __init__(self, network: torch.nn.Module, target: str, *, half: bool,
                 cache_dir: Path) -> None:  # fmt: skip
        ov = import_openvino()
        self.target = target
        self._core = ov.Core()
        with contextlib.suppress(Exception):  # only a speed-up for the next runs
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._core.set_property({"CACHE_DIR": str(cache_dir)})
        model = ov.convert_model(
            network, example_input=torch.zeros(1, 3, 64, 64),
            input=[ov.PartialShape([1, 3, -1, -1])],
        )  # fmt: skip
        self._model = _with_8bit_io(ov, model)
        self._config = {
            # FP16 is the GPU's fast path; CPUs without native support emulate it slowly.
            "INFERENCE_PRECISION_HINT": "f16" if half and target != "CPU" else "f32",
            "PERFORMANCE_HINT": "LATENCY",
        }
        if target == "CPU":
            # Same thread budget as PyTorch would get (per worker, see --torch-threads).
            self._config["INFERENCE_NUM_THREADS"] = torch.get_num_threads()
        self._compiled: dict[tuple[int, ...], object] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(image[..., ::-1])
        shape = rgb.shape
        requests = self._local.__dict__.setdefault("requests", {})
        if shape not in requests:
            requests[shape] = self._compile(shape).create_infer_request()
        request = requests[shape]
        request.infer({0: rgb[np.newaxis]})
        # Back to BGR. The request reuses its output buffer on the next call, so this copy
        # (which the swap needs anyway) is also what keeps the result.
        return np.ascontiguousarray(request.get_output_tensor(0).data[0][..., ::-1])

    def _compile(self, shape: tuple[int, ...]):
        with self._lock:
            if shape not in self._compiled:
                model = self._model.clone()
                model.reshape([1, *shape])
                self._compiled[shape] = self._core.compile_model(model, self.target, self._config)
            return self._compiled[shape]


def _with_8bit_io(ov, model):
    """``model`` (NCHW, floats in [0, 1]) made to take and return 8-bit NHWC images, in
    the same channel order (RGB): :class:`OpenVINOModel` swaps BGR ↔ RGB around it."""
    from openvino.preprocess import PrePostProcessor

    try:
        import openvino.opset13 as ops
    except ImportError:  # OpenVINO before the top-level opset modules
        import openvino.runtime.opset13 as ops

    def to_8bit(output):
        # Real-ESRGAN's own rounding: clamp to [0, 1], x255, round half to even.
        value = ops.clamp(output, 0.0, 1.0)
        value = ops.multiply(value, ops.constant(np.float32(255.0)))
        return ops.round(value, "half_to_even").output(0)

    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
    ppp.output().model().set_layout(ov.Layout("NCHW"))
    ppp.output().postprocess().custom(to_8bit)
    ppp.output().tensor().set_layout(ov.Layout("NHWC")).set_element_type(ov.Type.u8)
    return ppp.build()


def import_openvino():
    try:
        import openvino as ov
    except ImportError as exc:
        raise MissingDependencyError(
            "--device igpu and --backend openvino run the AI through OpenVINO. "
            f"Install it with: {INSTALL_OPENVINO}"
        ) from exc
    return ov
