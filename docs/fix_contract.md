# NVFP4 CroissAttn fix contract

The implementation is complete only when these invariants hold.

1. **Forward replay.** Quantize `Q`, `K`, and `V` once. Form the causal score
   from those payloads, compute the FP32 row maximum and denominator, quantize
   the masked numerator `U`, and form `O = (U_f / L)V_f`. Backward reuses the
   saved payloads, scales, shape and mask.
2. **Fixed auxiliary correction.** Recompute or retain `O' = P V_f` and set
   `D*_i = <dO_i, O'_i>`. Use `dS = P * (dP - D*) / sqrt(d)`; do not replace
   `D*` by a row sum of the sampled `dP`.
3. **Six FP4 products.** Use FP4 operands for `QK^T`, `PV`, `dO V^T`,
   `U^T(L^-1dO)`, `dS K`, and `dS^T Q`. Accumulate in FP32 or a higher
   precision supported by the target kernel.
4. **Consumer-axis recasts.** A saved factor is decoded and requantized when
   its contraction axis changes. NVFP4 block scales are always formed over
   the actual contraction axis in groups of 16.
5. **Independent SR.** Each backward product receives fresh stochastic draws
   for both operands. Forward payloads use nearest rounding and are never
   freshly re-quantized during score replay.
6. **Value normalization.** `dO` is divided by `L` before it is quantized in
   the `dV` product, matching `dV* = U_f^T(L^-1dO)`.
7. **Bounded execution.** Query tiles own `dQ` accumulation. `dK` and `dV`
   are accumulated in separate key tiles. No global `[S,S]` gradient tensor
   is required when `chunk_size` is set.

The reference backend intentionally exposes these steps in Python. A native
Blackwell implementation should preserve the same payload and scale metadata,
then replace only the `fp4_matmul` primitive and tile loops.

Projection modules follow the same type rules: forward inputs and weights are
saved as NVFP4 payloads, dgrad uses a weight recast along the output axis, and
wgrad uses an activation recast along the flattened batch axis. Their two
backward products use independent stochastic streams; autograd never silently
falls back to a decoded high precision weight for either product.
