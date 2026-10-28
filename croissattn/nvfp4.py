"""NVFP4 quantization and FP4 matrix products.

NVFP4 has an E2M1 payload, one positive E4M3 scale for every 16 values, and
one FP32 tensor scale.  The code here keeps the payload as an unpacked code
tensor so the algorithm remains portable and easy to audit.  A production
Blackwell backend can replace :func:`fp4_matmul` without changing the
attention contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0

# E2M1 has a shared zero and seven non-zero magnitudes.  Keeping the codebook
# explicit avoids silently using an FP8 cast for an FP4 operation.
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_POSITIVE = torch.tensor(_FP4_LEVELS, dtype=torch.float32)
_E4M3_POSITIVE = torch.tensor(
    [
        2.0 ** (e - 7) * (1.0 + m / 8.0)
        for e in range(16)
        for m in range(8)
        if e != 0 and 2.0 ** (e - 7) * (1.0 + m / 8.0) <= 448.0
    ],
    dtype=torch.float32,
)


def _device_codebook(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return _FP4_POSITIVE.to(device=device), _E4M3_POSITIVE.to(device=device)


def _axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is invalid for a {ndim}-D tensor")
    return axis % ndim


def _ceil_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round positive values upward to a finite positive E4M3 value."""

    _, levels = _device_codebook(x.device)
    levels = levels.unique(sorted=True)
    finite = torch.isfinite(x)
    safe = torch.where(finite, x, torch.full_like(x, float("inf")))
    idx = torch.searchsorted(levels, safe.clamp_min(float(levels[0])))
    idx = idx.clamp_max(levels.numel() - 1)
    out = levels[idx]
    return torch.where(x > 0, out, torch.ones_like(out))


def _fp4_quantize_values(
    x: torch.Tensor, *, stochastic: bool, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """Quantize values whose representable range is already [-6, 6]."""

    levels, _ = _device_codebook(x.device)
    negative = x < 0
    ax = x.abs().clamp_max(FP4_MAX)
    upper = torch.searchsorted(levels, ax).clamp_max(levels.numel() - 1)
    lower = (upper - 1).clamp_min(0)
    lo = levels[lower]
    hi = levels[upper]
    # Exact codebook values should stay exact instead of being treated as a
    # stochastic interval with a zero-width denominator.
    frac = torch.where(hi > lo, (ax - lo) / (hi - lo), torch.zeros_like(ax))
    if stochastic:
        if generator is None:
            rnd = torch.rand(ax.shape, device=ax.device, dtype=ax.dtype)
        else:
            rnd = torch.rand(
                ax.shape, device=ax.device, dtype=ax.dtype, generator=generator
            )
        mag = torch.where(rnd < frac, hi, lo)
    else:
        mag = torch.where(frac < 0.5, lo, hi)
    # E2M1 uses three magnitude bits and one sign bit.  Zero has a canonical
    # positive code, so signed zero cannot create a second representation.
    # searchsorted is the inverse of the codebook selection above.  Recompute
    # from the selected magnitude to avoid retaining a large one-hot tensor.
    magnitude_code = torch.searchsorted(levels, mag).clamp_max(levels.numel() - 1)
    return (magnitude_code.to(torch.uint8) | (negative & (mag > 0)).to(torch.uint8) * 8)


def _fresh_generator(device: torch.device) -> torch.Generator:
    """Create a new stream for one independent operand draw."""

    generator = torch.Generator(device=device)
    # torch.seed() consumes the process RNG and gives every recast a distinct
    # stream.  This is deliberately done once per operand, never per element.
    generator.manual_seed(torch.seed())
    return generator


@dataclass
class NVFP4Tensor:
    """Unpacked NVFP4 payload and its hierarchical scales.

    ``axis`` is the axis along which block scales were formed.  Keeping it in
    the object is essential: a saved factor must be recast when a later GEMM
    uses a different contraction axis.
    """

    payload: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor
    original_shape: tuple[int, ...]
    axis: int
    padded: int

    def decode(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        codes = self.payload.to(torch.long)
        levels, _ = _device_codebook(self.payload.device)
        moved = levels[codes & 0x7]
        moved = torch.where(
            (codes & 0x8).bool() & ((codes & 0x7) != 0), -moved, moved
        )
        scales = self.block_scale.to(torch.float32).repeat_interleave(
            BLOCK_SIZE, dim=-1
        )[..., : self.padded]
        moved = moved * scales * self.global_scale.to(torch.float32)
        moved = moved[..., : self.original_shape[self.axis]]
        out = moved.movedim(-1, self.axis)
        return out.to(dtype=dtype)

    def recast(self, axis: int, *, stochastic: bool = True) -> "NVFP4Tensor":
        """Decode and quantize for a new consumer contraction axis."""

        return quantize_nvfp4(self.decode(), axis=axis, stochastic=stochastic)

    def packed_payload(self) -> torch.Tensor:
        """Pack two E2M1 codes per byte for a native NVFP4 kernel.

        The returned tensor keeps the quantizer's moved layout (the block axis
        is last), which is the layout expected by Blackwell FP4 GEMM adapters.
        ``block_scale`` and ``global_scale`` remain separate metadata.
        """

        if self.padded % 2:
            raise ValueError("NVFP4 payload width must be even to pack nibbles")
        codes = self.payload.contiguous()
        return codes[..., 0::2] | (codes[..., 1::2] << 4)


def quantize_nvfp4(
    x: torch.Tensor,
    *,
    axis: int = -1,
    stochastic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> NVFP4Tensor:
    """Quantize ``x`` to hierarchical NVFP4.

    Forward operands use nearest rounding.  Backward operands pass
    ``stochastic=True`` and receive an independent stream unless a generator
    is explicitly supplied by a deterministic test.
    """

    if not x.is_floating_point():
        raise TypeError("NVFP4 quantization expects a floating point tensor")
    axis = _axis(axis, x.ndim)
    moved = x.detach().float().movedim(axis, -1)
    width = moved.shape[-1]
    padded = ((width + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded != width:
        moved = torch.nn.functional.pad(moved, (0, padded - width))
    flat = moved.reshape(-1, padded)
    amax = flat.abs().amax()
    tiny = torch.finfo(torch.float32).tiny
    global_scale = torch.clamp(amax / (FP4_MAX * FP8_MAX), min=tiny)
    blocks = flat.reshape(-1, padded // BLOCK_SIZE, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)
    target = block_amax / (FP4_MAX * global_scale)
    block_scale = _ceil_e4m3(target)
    # A zero block still has a valid positive scale; its payload is zero.
    normalized = flat / (
        global_scale * block_scale.repeat_interleave(BLOCK_SIZE, dim=-1)
    )
    if stochastic and generator is None:
        generator = _fresh_generator(x.device)
    payload = _fp4_quantize_values(
        normalized,
        stochastic=stochastic,
        generator=generator,
    ).reshape_as(moved)
    # Store the code values rather than decoded FP32.  This keeps the replay
    # state explicit even though the portable backend does not pack nibbles.
    block_scale = block_scale.reshape(*moved.shape[:-1], padded // BLOCK_SIZE)
    return NVFP4Tensor(
        payload=payload,
        block_scale=block_scale,
        global_scale=global_scale,
        original_shape=tuple(x.shape),
        axis=axis,
        padded=padded,
    )


def fp4_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    a_axis: int = -1,
    b_axis: int = -2,
    stochastic: bool = True,
) -> torch.Tensor:
    """Run one FP4-operand GEMM with higher-precision accumulation.

    The axes default to the contraction axes of ``[..., M, K]`` and
    ``[..., K, N]``.  Both operands are independently quantized, including
    fresh stochastic streams, as required by the CroissAttn estimator.
    """

    aq = quantize_nvfp4(a, axis=a_axis, stochastic=stochastic)
    bq = quantize_nvfp4(b, axis=b_axis, stochastic=stochastic)
    return torch.matmul(aq.decode(), bq.decode())


__all__ = [
    "BLOCK_SIZE",
    "FP4_MAX",
    "FP8_MAX",
    "NVFP4Tensor",
    "fp4_matmul",
    "quantize_nvfp4",
]
