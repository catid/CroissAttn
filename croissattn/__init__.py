"""CroissAttn NVFP4 training primitives.

The package is intentionally independent of Transformer Engine.  It provides
an auditable reference implementation that can run on CPU or CUDA and a
straightforward integration point for replacing the matrix products with a
Blackwell NVFP4 kernel.
"""

from .attention import NVFP4Attention, attention, backward_components
from .linear import NVFP4Linear
from .nvfp4 import NVFP4Tensor, quantize_nvfp4

__all__ = [
    "NVFP4Attention",
    "NVFP4Linear",
    "NVFP4Tensor",
    "attention",
    "backward_components",
    "quantize_nvfp4",
]
