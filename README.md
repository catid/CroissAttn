# CroissAttn NVFP4

This repository contains a source-visible NVFP4 attention implementation that
implements the complete CroissAttn forward and backward contract.  It is an
auditable reference and an integration point for a native Blackwell kernel;
the portable Python/Torch path is not a claim of production throughput.

The forward quantizes Q, K, V and the masked softmax numerator U to NVFP4
(E2M1 payloads, E4M3 scale per 16 values, FP32 tensor scale).  It saves those
payloads and replays the exact quantized score path in backward.  The backward
has all six low precision products:

* `Q_f K_f^T` and `P_f V_f` in forward;
* `dO V_f^T`, `U_f^T (L^-1 dO)`, `dS K_f`, and `dS^T Q_f` in backward.

The implementation includes the full fix set from the CroissAttn estimator:

* fixed auxiliary output `O' = P V_f` and `D* = <dO, O'>` correction;
* exact saved Q/K/V/U payload replay and causal mask replay;
* independent stochastic rounding streams for each backward operand;
* consumer-axis recasts with block scales regrouped on the contraction axis;
* normalization of `dO` before the `dV` product;
* FP32 softmax, row reductions and accumulators;
* query-row-owned tiled accumulation with `chunk_size` to bound workspace.

```python
import torch
from croissattn import attention

q = torch.randn(2, 16, 2048, 128, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = attention(q, k, v, chunk_size=256)
out.float().square().mean().backward()
```

`NVFP4Linear` supplies a projection module with saved forward payloads and
explicit NVFP4 dgrad/wgrad products. Each backward operand is independently
stochastically rounded and recast to its consumer contraction axis. Replace
the `torch.matmul` calls in `nvfp4.py` with the device's native NVFP4 GEMM to
build a complete high-throughput trainer.

Run the checks with `pip install -e .` followed by `pytest -q`.  On the GB300
reference container, the CUDA smoke test covers `B=1,H=2,S=16,D=16` and
chunk sizes 4, 8, and full sequence; every output and gradient is finite. The
portable path is intentionally slower than TE's native NVFP4 MLP kernels until
the `fp4_matmul` hook is replaced by a block-scaled Blackwell GEMM.

The repository was initialized from an empty `catid/CroissAttn` remote.  The
SSH remote is not authenticated in this environment, so the local commit is
ready to push once GitHub credentials are available.
