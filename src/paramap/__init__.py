"""
ParamMap: budgeted, lossless VRAM/RAM residency for local LLMs.

Keeps every parameter of a model but holds only a budgeted subset in VRAM at any
moment, streaming the rest from host RAM over PCIe. Output is identical to
running the model fully on the GPU; what you trade is generation speed for VRAM
headroom.

The archived, lossy domain-conditioned approach lives in `paramap.legacy`.
"""

from .residency import (
    ResidencyPlan,
    StreamingModel,
    plan_residency,
)

__version__ = "0.2.0"
__all__ = ["ResidencyPlan", "StreamingModel", "plan_residency"]
