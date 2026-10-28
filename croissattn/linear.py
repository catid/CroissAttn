"""A small NVFP4 linear layer for end-to-end trainer experiments."""

from __future__ import annotations

import torch
from torch import nn

from .nvfp4 import quantize_nvfp4


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 forward operands and an STE backward.

    The attention module in :mod:`croissattn.attention` implements the full
    stochastic CroissAttn backward.  For projections, this layer uses native
    ``torch.matmul`` on decoded NVFP4 operands and an identity straight-through
    estimator for the quantizer.  A Blackwell extension can replace the two
    decoded matmuls with ``nvte_cublas_gemm`` while retaining this API.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        wq = quantize_nvfp4(self.weight, axis=-1, stochastic=False).decode(
            dtype=input.dtype
        )
        # The STE keeps optimizer gradients connected to the master parameter;
        # the forward value is still exactly the decoded NVFP4 weight.
        w = wq + (self.weight - self.weight.detach())
        xq = quantize_nvfp4(input, axis=-1, stochastic=False).decode(dtype=input.dtype)
        x = xq + (input - input.detach())
        out = torch.matmul(x, w.transpose(-2, -1))
        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = ["NVFP4Linear"]
