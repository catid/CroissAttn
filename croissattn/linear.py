"""A small NVFP4 linear layer for end-to-end trainer experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from .nvfp4 import fp4_matmul, quantize_nvfp4


class _NVFP4LinearFunction(Function):
    """Projection path with saved NVFP4 payloads and FP4 backward GEMMs.

    The ordinary STE implementation is useful as a fallback, but it lets
    autograd differentiate through decoded BF16/FP32 tensors.  This function
    keeps the forward payloads and explicitly recasts each saved operand for
    its backward contraction axis, matching the CroissAttn estimator's type
    contract.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias):
        if input.ndim < 1 or weight.ndim != 2:
            raise ValueError("NVFP4Linear expects input [..., in_features] and 2D weight")
        if input.shape[-1] != weight.shape[-1]:
            raise ValueError("input last dimension must equal weight in_features")
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError("bias must have shape [out_features]")

        # Forward uses nearest rounding once.  The saved objects retain the
        # original block-axis metadata for backward consumer-axis recasts.
        x_f = quantize_nvfp4(input, axis=-1, stochastic=False)
        w_f = quantize_nvfp4(weight, axis=-1, stochastic=False)
        x_dec = x_f.decode(dtype=torch.float32)
        w_dec = w_f.decode(dtype=torch.float32)
        output = torch.matmul(x_dec, w_dec.transpose(-2, -1))
        if bias is not None:
            output = output + bias.float()

        ctx.x_f = x_f
        ctx.w_f = w_f
        ctx.input_shape = tuple(input.shape)
        ctx.input_dtype = input.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.has_bias = bias is not None
        return output.to(dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_dec = ctx.x_f.decode(dtype=torch.float32)
        w_dec = ctx.w_f.decode(dtype=torch.float32)
        grad_output = grad_output.float()

        # dX = dY W_f.  W was saved along its input axis, so fp4_matmul
        # explicitly recasts it along the output/contraction axis here.
        grad_input = fp4_matmul(
            grad_output,
            w_dec,
            a_axis=-1,
            b_axis=-2,
            stochastic=True,
        )

        # dW = dY^T X_f.  Flatten all batch/sequence dimensions into the
        # contraction axis and recast X from its forward feature axis.
        n = grad_output.numel() // grad_output.shape[-1]
        grad_y_2d = grad_output.reshape(n, grad_output.shape[-1])
        x_2d = x_dec.reshape(n, x_dec.shape[-1])
        grad_weight = fp4_matmul(
            grad_y_2d.transpose(0, 1),
            x_2d,
            a_axis=-1,
            b_axis=-2,
            stochastic=True,
        )
        grad_bias = grad_output.reshape(n, grad_output.shape[-1]).sum(dim=0) if ctx.has_bias else None
        return (
            grad_input.to(ctx.input_dtype).reshape(ctx.input_shape),
            grad_weight.to(ctx.weight_dtype),
            None if grad_bias is None else grad_bias.to(ctx.bias_dtype),
        )


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 forward and backward operands.

    The attention module in :mod:`croissattn.attention` implements the full
    stochastic CroissAttn backward.  For projections, this layer saves forward
    NVFP4 payloads and uses independently stochastic, consumer-axis recast FP4
    dgrad and wgrad products. A Blackwell extension can replace the decoded
    matmuls with ``nvte_cublas_gemm`` while retaining this API.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _NVFP4LinearFunction.apply(input, self.weight, self.bias)


__all__ = ["NVFP4Linear"]
