"""Tropical Bellman-Ford model (blueprint §5.1, simplified d=1 first version).

For each graph in the batch we hold a tropical operator A ∈ R_max^{N x N}
produced by a small edge hypernetwork from raw edge features (the weight
itself, plus presence indicator). The dynamics are:

    h^{(0)}_i = 0 if i = source, else -inf  (max-plus source indicator)
    h^{(t+1)} = A^T ⊗ h^{(t)}              (one tropical message-passing step)

We use A^T because (A^T ⊗ h)_i = max_j (A_{ji} + h_j) = max-weight incoming
walk to i, which is the *correct* update for max-weight paths ending at i.
(With self-loops at weight 0 and missing edges at -inf, this collapses to
the usual max-plus Bellman update.)

Two forward modes:
    forward(..., n_steps=T)  -> iterative: h <- A^T ⊗ h, T times. With tropical
                                accumulation we keep max(h^{(t)}) componentwise.
    forward(..., one_shot=True) -> one matvec against the Kleene star:
                                h_inf = A^{*,T} ⊗ source_indicator.

When T >= N-1, both modes return the same answer (Lemma 4: Kleene collapse).

Currently d = 1 (one scalar per node). A learnable invertible flow g over the
*per-node* state can be plugged in later (just wrap each h_i through g and
swap A for a block-tropical operator) - see §5.1 of the blueprint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.bellman_ford import MISSING_EDGE
from ..tropical.kleene import kleene_squaring
from ..tropical.ops import NEG_INF, soft_trop_matvec, st_trop_matvec, trop_matvec


def _ensure_mode(mode: str) -> None:
    if mode not in ("hard", "soft", "st"):
        raise ValueError(f"Unknown mode {mode!r} (choose hard|soft|st)")


class EdgeHypernet(nn.Module):
    """MLP from per-edge features to a scalar tropical weight A_{ij}.

    Input per edge: 2-vector (w_{ij}, presence_{ij}) where w_{ij} is the raw
    min-plus weight (0 if no edge, treated as a feature) and presence is 1.0
    if the edge exists, else 0.0. The hypernetwork outputs a real-valued
    tropical entry; missing edges are overridden to NEG_INF outside.

    For min-plus shortest-path problems the "right" answer is A_{ij} = -w_{ij}
    (so max-plus paths on A correspond to shortest paths on W). The MLP can
    learn this exactly; we use a 2-layer net so it has just enough capacity
    to discover the negation + scale + bias.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer so the residual is 0 at start; the
        # `-alpha * w + beta` skip in the caller then makes A = -W at init.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, e_feats: torch.Tensor) -> torch.Tensor:
        """e_feats: (..., 2) -> (...,) tropical scalar."""
        return self.net(e_feats).squeeze(-1)


class BellmanFordTropical(nn.Module):
    """Tropical Linearizer for SSSP / all-pairs Bellman-Ford.

    Identity g (d=1 per node, no invertible flow). The "tropical core" is the
    edge-feature hypernetwork that produces the N x N max-plus matrix A.
    """

    def __init__(self, hidden: int = 16, learn_skip: bool = True):
        super().__init__()
        self.hyper = EdgeHypernet(hidden=hidden)
        # Skip parameter: A_{ij} = -alpha * w_{ij} + beta + hyper(features).
        # alpha starts at 1 (so the network is initialised at the correct
        # solution A = -W), and is learnable so it can adjust if features
        # come in different units.
        if learn_skip:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("alpha", torch.tensor(1.0))
            self.register_buffer("beta", torch.tensor(0.0))

    # ---- build the tropical operator A from edge weights ----

    def edge_features(self, W: torch.Tensor) -> torch.Tensor:
        """Build per-edge feature tensor of shape (B, N, N, 2)."""
        presence = (W < MISSING_EDGE / 10).float()
        # Mask out the raw weight in missing-edge positions so the MLP doesn't
        # see the huge MISSING_EDGE sentinel.
        w = torch.where(presence > 0.5, W, torch.zeros_like(W))
        return torch.stack([w, presence], dim=-1)

    def build_A(self, W: torch.Tensor) -> torch.Tensor:
        """W: (B, N, N) min-plus weights. Returns A: (B, N, N) max-plus tropical."""
        feats = self.edge_features(W)
        presence = feats[..., 1]
        raw_w = feats[..., 0]
        # Skip + residual: A_init = -alpha*w + beta on edges, NEG_INF off-edges.
        A_skip = -self.alpha * raw_w + self.beta
        A_residual = self.hyper(feats)
        A = A_skip + A_residual
        # Force the additive identity (NEG_INF) where the edge is absent.
        A = torch.where(presence > 0.5, A, torch.full_like(A, NEG_INF))
        return A

    # ---- forward modes ----

    def _source_indicator(self, source: torch.Tensor, n: int) -> torch.Tensor:
        """Build a max-plus indicator vector with 0 at the source, -inf elsewhere."""
        B = source.shape[0]
        ind = torch.full((B, n), NEG_INF, dtype=torch.float32, device=source.device)
        ind.scatter_(1, source.unsqueeze(1), 0.0)
        return ind

    def _matvec(self, A: torch.Tensor, x: torch.Tensor, *, mode: str, beta: float | None) -> torch.Tensor:
        if mode == "hard":
            return trop_matvec(A, x)
        if mode == "soft":
            assert beta is not None
            return soft_trop_matvec(A, x, beta=beta)
        if mode == "st":
            assert beta is not None
            return st_trop_matvec(A, x, beta=beta)
        raise ValueError(mode)

    def forward(
        self,
        W: torch.Tensor,
        source: torch.Tensor,
        *,
        n_steps: int | None = None,
        one_shot: bool = False,
        mode: str = "hard",
        beta: float | None = None,
        return_A: bool = False,
    ):
        """Run the tropical Bellman-Ford.

        Args:
            W: (B, N, N) min-plus edge weights.
            source: (B,) integer source node per graph.
            n_steps: number of iterative tropical-MPNN steps. Ignored if one_shot.
            one_shot: if True, compute A* once and apply it to the source indicator.
            mode: "hard" (exact max), "soft" (logsumexp/β), "st" (straight-through).
            beta: temperature for soft/st modes.
            return_A: also return the tropical operator A (for diagnostics).

        Returns: predicted max-plus shortest paths y of shape (B, N).
                 Min-plus distances are -y (with NEG_INF translated to MISSING_EDGE).
        """
        _ensure_mode(mode)
        n = W.shape[-1]
        A = self.build_A(W)
        # A is N×N with A_{ji} = max-plus weight of edge j -> i (post-hyper).
        # The Bellman update is h^{(t+1)}_i = max_j (A_{ji} + h^{(t)}_j),
        # i.e. trop_matvec(A^T, h).
        A_T = A.transpose(-1, -2)
        h0 = self._source_indicator(source, n)

        if one_shot:
            # F_inf(x) = A^{*,T} ⊗ x where x is the source indicator.
            A_T_star = kleene_squaring(A_T, n_iter=None, mode=mode, beta=beta)
            y = self._matvec(A_T_star, h0, mode=mode, beta=beta)
        else:
            assert n_steps is not None, "Need n_steps when not one_shot"
            h = h0
            agg = h0.clone()
            for _ in range(n_steps):
                h = self._matvec(A_T, h, mode=mode, beta=beta)
                agg = torch.maximum(agg, h)
            y = agg

        if return_A:
            return y, A
        return y


def predicted_distances(y_max_plus: torch.Tensor) -> torch.Tensor:
    """Convert a max-plus output y (B, N) to min-plus distances (with
    NEG_INF -> MISSING_EDGE so loss masking works)."""
    d = -y_max_plus
    d = torch.where(d > MISSING_EDGE / 10, torch.full_like(d, MISSING_EDGE), d)
    return d
