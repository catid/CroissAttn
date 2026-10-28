"""Complete CroissAttn NVFP4 attention forward and backward.

The implementation follows the paper's execution contract rather than the
usual ``softmax(QK^T)V`` autograd graph:

* Q, K, V and the unnormalised numerator U are quantized once in forward and
  their payloads are replayed in backward;
* the forward auxiliary ``O' = P V_f`` and fixed ``D* = <dO, O'>`` are used in
  the softmax correction;
* all six matrix products use independently quantized FP4 operands;
* saved factors are decoded and recast along each consumer's contraction axis;
* softmax, row reductions and accumulators stay in FP32.

The Python backend is a correctness and integration reference.  ``chunk_size``
keeps the backward products bounded and is the hook used by a fused QRO CUDA
kernel.  It does not claim the throughput of a native Blackwell kernel.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from .nvfp4 import fp4_matmul, quantize_nvfp4


def _causal_mask(sq: int, sk: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((sq, sk), dtype=torch.bool, device=device), diagonal=1
    )


def _score_replay(qf: torch.Tensor, kf: torch.Tensor, scale: float) -> torch.Tensor:
    """Replay the saved forward score using decoded saved payloads."""

    return torch.matmul(qf, kf.transpose(-2, -1)) * scale


def backward_components(
    qf,
    kf,
    vf,
    uf,
    p: torch.Tensor,
    l: torch.Tensor,
    do: torch.Tensor,
    *,
    scale: float,
    chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the complete stochastic CroissAttn backward.

    Returns ``(dQ, dK, dV, Dstar)``.  ``Dstar`` is returned for tests and
    instrumentation so callers can verify that the fixed auxiliary correction
    is used instead of a sampled row sum.
    """

    q_dec = qf.decode()
    k_dec = kf.decode()
    v_dec = vf.decode()
    u_dec = uf.decode()
    # The auxiliary uses the unquantized softmax probabilities and the saved
    # forward V payload.  It is computed before any stochastic backward draw.
    o_prime = torch.matmul(p, v_dec)
    dstar = (do.float() * o_prime.float()).sum(dim=-1, keepdim=True)

    sq, sk = p.shape[-2:]
    tile = max(int(chunk_size), 1) if chunk_size else max(sq, sk)
    dn = do / l.to(dtype=do.dtype)

    dq = torch.zeros_like(q_dec)
    dk = torch.zeros_like(k_dec)
    dv = torch.zeros_like(v_dec)

    # One tiled QRO traversal computes a single dP/dS tile and feeds both
    # dependent score products.  This preserves the paper's dependency order:
    # dQ and dK see the same sampled dS tile, while each GEMM gets independent
    # operand recasts and an independent stochastic stream.
    for q0 in range(0, sq, tile):
        q1 = min(q0 + tile, sq)
        do_i = do[..., q0:q1, :]
        dstar_i = dstar[..., q0:q1, :]
        dn_i = dn[..., q0:q1, :]
        for k0 in range(0, sk, tile):
            k1 = min(k0 + tile, sk)
            p_ij = p[..., q0:q1, k0:k1]
            v_j = v_dec[..., k0:k1, :]
            dp_ij = fp4_matmul(
                do_i,
                v_j.transpose(-2, -1),
                a_axis=-1,
                b_axis=-2,
                stochastic=True,
            )
            ds_ij = p_ij * (dp_ij.float() - dstar_i.float()) * scale
            dq[..., q0:q1, :] += fp4_matmul(
                ds_ij,
                k_dec[..., k0:k1, :],
                a_axis=-1,
                b_axis=-2,
                stochastic=True,
            ).to(dq.dtype)
            dk[..., k0:k1, :] += fp4_matmul(
                ds_ij.transpose(-2, -1),
                q_dec[..., q0:q1, :],
                a_axis=-1,
                b_axis=-2,
                stochastic=True,
            ).to(dk.dtype)
            # U^T (L^-1 dO) is accumulated by key tile.  The normalization is
            # applied before quantization, as required by the dV identity.
            u_ij = u_dec[..., q0:q1, k0:k1]
            dv[..., k0:k1, :] += fp4_matmul(
                u_ij.transpose(-2, -1),
                dn_i,
                a_axis=-1,
                b_axis=-2,
                stochastic=True,
            ).to(dv.dtype)

    return dq, dk, dv, dstar


class _NVFP4AttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: Optional[float],
        chunk_size: Optional[int],
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k and v must have shape [batch, heads, seq, dim]")
        if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
            raise ValueError("q, k and v must have matching batch and head axes")
        if q.shape[-1] != k.shape[-1] or v.shape[-1] != q.shape[-1]:
            raise ValueError("q, k and v must have the same head dimension")
        if q.shape[-2] != k.shape[-2] or v.shape[-2] != k.shape[-2]:
            raise ValueError("the reference path currently requires self-attention")
        if q.device != k.device or q.device != v.device:
            raise ValueError("q, k and v must be on the same device")

        head_dim = q.shape[-1]
        attn_scale = float(scale) if scale is not None else head_dim**-0.5
        # Forward layout: Q/K contract on D, V contracts with P on sequence,
        # and U contracts with V on sequence.  These axes are part of the
        # replay state and must not be silently changed in backward.
        qf = quantize_nvfp4(q, axis=-1, stochastic=False)
        kf = quantize_nvfp4(k, axis=-1, stochastic=False)
        vf = quantize_nvfp4(v, axis=-2, stochastic=False)
        q_dec, k_dec, v_dec = qf.decode(), kf.decode(), vf.decode()
        scores = _score_replay(q_dec, k_dec, attn_scale).float()
        scores = scores.masked_fill(
            _causal_mask(scores.shape[-2], scores.shape[-1], scores.device),
            float("-inf"),
        )
        p = torch.softmax(scores, dim=-1)
        row_max = scores.amax(dim=-1, keepdim=True)
        u = torch.exp(scores - row_max)
        l = u.sum(dim=-1, keepdim=True)
        # Masked entries are exactly zero before quantization and remain zero
        # in the replay payload.
        uf = quantize_nvfp4(u, axis=-1, stochastic=False)
        u_dec = uf.decode()
        output = (torch.matmul(u_dec, v_dec) / l).to(dtype=q.dtype)

        ctx.qf, ctx.kf, ctx.vf, ctx.uf = qf, kf, vf, uf
        ctx.p, ctx.l = p, l
        ctx.attn_scale = attn_scale
        ctx.chunk_size = chunk_size
        ctx.input_dtype = q.dtype
        return output

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        dq, dk, dv, _ = backward_components(
            ctx.qf,
            ctx.kf,
            ctx.vf,
            ctx.uf,
            ctx.p,
            ctx.l,
            do,
            scale=ctx.attn_scale,
            chunk_size=ctx.chunk_size,
        )
        return dq.to(ctx.input_dtype), dk.to(ctx.input_dtype), dv.to(ctx.input_dtype), None, None


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply causal self-attention with complete NVFP4 forward/backward."""

    return _NVFP4AttentionFunction.apply(q, k, v, scale, chunk_size)


class NVFP4Attention(nn.Module):
    """Drop-in module for tensors shaped ``[B, H, S, D]``."""

    def __init__(self, head_dim: int, *, chunk_size: Optional[int] = None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.chunk_size = chunk_size

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return attention(q, k, v, chunk_size=self.chunk_size)


__all__ = ["NVFP4Attention", "attention", "backward_components"]
