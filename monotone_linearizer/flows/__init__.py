"""Re-export the VectorFlow from the tropical_linearizer package.

The flow architecture (ActNorm1d + InvLinear + AffineCoupling1d + Permutation,
K-block stack) was implemented and exhaustively tested in
tropical_linearizer/flows/vector_flow.py during the previous research thread.
We re-use it directly rather than reimplementing.
"""

from tropical_linearizer.flows.vector_flow import (
    ActNorm1d,
    AffineCoupling1d,
    InvLinear,
    VectorCouplingBlock,
    VectorFlow,
)

__all__ = [
    "ActNorm1d",
    "InvLinear",
    "AffineCoupling1d",
    "VectorCouplingBlock",
    "VectorFlow",
]
