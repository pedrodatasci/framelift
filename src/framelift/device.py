"""Choosing where the AI runs (CUDA GPU or CPU) and tuning PyTorch for it."""

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass

import torch

from .errors import InvalidOptionError


@dataclass(frozen=True)
class Device:
    """The compute device a run will use."""

    kind: str
    """``"cuda"`` or ``"cpu"``."""

    name: str
    """Human-friendly name, e.g. ``"NVIDIA GeForce RTX 3060"`` or ``"CPU"``."""

    gpu_id: int | None = None

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def torch_device(self) -> torch.device:
        return torch.device(f"cuda:{self.gpu_id}") if self.is_cuda else torch.device("cpu")


def resolve_device(requested: str, gpu_id: int = 0) -> Device:
    """Turn ``auto`` / ``cpu`` / ``cuda`` into a concrete :class:`Device`."""
    if requested not in ("auto", "cpu", "cuda"):
        raise InvalidOptionError("device must be auto, cpu or cuda.")

    cuda_available = torch.cuda.is_available()

    if requested == "cuda" and not cuda_available:
        raise InvalidOptionError(
            "You asked for --device cuda, but PyTorch can't see a CUDA GPU. "
            "On Intel/AMD graphics or machines without NVIDIA drivers, use --device cpu "
            "(or install a CUDA-enabled PyTorch build if you do have an NVIDIA card)."
        )

    if requested == "cpu" or not cuda_available:
        return Device(kind="cpu", name="CPU")

    return Device(kind="cuda", name=torch.cuda.get_device_name(gpu_id), gpu_id=gpu_id)


def configure_torch(cpu_threads: int = 0) -> None:
    """Apply process-wide PyTorch performance settings.

    * ``cpu_threads > 0`` pins PyTorch's CPU thread pool size.
    * On CUDA, enables cuDNN autotuning and TF32 math (much faster on RTX
      30xx and newer, with no visible quality difference for this task).
    """
    if cpu_threads and cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        # Can only be set once per process; harmless if it was already set.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass  # very old PyTorch builds don't have these switches


def gpu_memory_gb(device: Device) -> float | None:
    """Total memory of the device's GPU in GB, or None on CPU."""
    if not device.is_cuda:
        return None
    return torch.cuda.get_device_properties(device.gpu_id).total_memory / 1024**3


def current_cpu_threads() -> int:
    return torch.get_num_threads()


def synchronize(device: Device) -> None:
    """Wait for queued GPU work to finish, so timings measure real work."""
    if device.is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(device.gpu_id)


def release_memory(device: Device) -> None:
    """Run Python's GC and give cached GPU memory back to the driver."""
    gc.collect()
    if device.is_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
