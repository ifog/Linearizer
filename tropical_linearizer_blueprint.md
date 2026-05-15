# Tropical Linearizer — Paper & Implementation Blueprint

**Working title:** *Tropical Linearizers: Learned Coordinates for Exact (max, +)-Linear Neural Networks*

**Target venue:** ICLR 2027 (primary) or NeurIPS 2026 if timing permits.

**Type:** Mostly empirical follow-up to the Linearizer paper (Shocher, Berman, Hallak — under review at ICLR 2026), extending the framework from the linear semiring $(\mathbb{R}, +, \times)$ to the tropical semiring $(\mathbb{R}_{\max}, \max, +)$.

**Owner / lead:** Assaf Hallak. Collaborators TBD — see §13.

**Status:** Blueprint v0. Hand off to Claude Code for both writing scaffold and implementation.

---

## 0. Quick orientation — what this paper is and isn't

**Core claim.** A learned invertible coordinate map $g$ can transport a conventionally non-tropical neural network into a space where it is *exactly* tropical-linear, i.e.
$$f(x) = g_y^{-1}\bigl(A \otimes g_x(x)\bigr),\quad \text{where } (A \otimes z)_i = \max_j (A_{ij} + z_j).$$
This inherits the entire algebraic machinery of the original Linearizer — composition closure, decomposition theory, idempotency-by-construction — but in the tropical setting, which is the natural algebra of shortest paths, dynamic programming, and piecewise-linear projection.

**What's new vs. the original Linearizer.**
- Different semiring → different structural primitives → different applications.
- Kleene star replaces matrix power; tropical idempotency replaces Euclidean projection; tropical polytopes replace linear subspaces.

**What's new vs. existing tropical NN work.** Existing tropical/morphological NNs use $(\max, +)$ as an architectural primitive in *standard* coordinates. We are the first to:
1. Use a *learned invertible coordinate transport* that makes a non-tropical mapping exactly tropical-linear in induced coordinates.
2. Architecturally enforce tropical-algebraic constraints (Kleene-star structure, tropical idempotency) on the core operator and prove they transfer to $f$.
3. Demonstrate that the *N-step collapse* from the original Linearizer extends to dynamic-programming-like recurrences via the Kleene star, giving one-shot tropical reasoning.

**What this paper does NOT claim.**
- Not a universal-approximation result. Expressivity of $f = g^{-1} \circ A_{\text{trop}} \circ g$ is bounded by tropical-linear maps composed with diffeomorphisms; we discuss what this excludes in §10.
- Not a pseudoinverse / inversion analog. The Cuninghame-Green tropical inverse exists only under restrictive conditions; we do not transfer the inversion application from the Euclidean paper. This is explicit in Limitations.
- Not a faster Bellman-Ford. We compete on *learnable, end-to-end-differentiable* shortest-path tasks (perception → path, OOD generalization), not on classical solver speed.

---

## 1. Notation and tropical preliminaries

We work in the max-plus semiring $\mathbb{R}_{\max} = (\mathbb{R} \cup \{-\infty\}, \oplus, \otimes)$ with:
- $a \oplus b := \max(a, b)$ — tropical addition.
- $a \otimes b := a + b$ — tropical multiplication.
- $\mathbf{0} := -\infty$ — additive identity.
- $\mathbf{1} := 0$ — multiplicative identity.

**Conventions:**
- Shortest-path tasks: implementation uses min-plus (negate or use `min` directly). The framework is symmetric under the isomorphism $a \mapsto -a$.
- All vectors live in $\mathbb{R}_{\max}^n$; $-\infty$ entries are implemented as a large negative constant (e.g. $-10^9$) with no-grad clamping. See §6 for numerics.

**Tropical matrix-vector product:** For $A \in \mathbb{R}_{\max}^{m \times n}$, $x \in \mathbb{R}_{\max}^n$:
$$(A \otimes x)_i = \bigoplus_{j=1}^n A_{ij} \otimes x_j = \max_j (A_{ij} + x_j).$$

**Tropical matrix-matrix product:**
$$(A \otimes B)_{ij} = \max_k (A_{ik} + B_{kj}).$$

**Tropical powers:** $A^{\otimes 0} = I$ (tropical identity: 0 on diagonal, $-\infty$ off-diagonal), $A^{\otimes k+1} = A \otimes A^{\otimes k}$.

**Kleene star.** For $A \in \mathbb{R}_{\max}^{n \times n}$, define
$$A^{*,N} := \bigoplus_{k=0}^N A^{\otimes k} = I \oplus A \oplus A^{\otimes 2} \oplus \dots \oplus A^{\otimes N}.$$

**Stabilization.** If $A$ has no positive-weight cycles (in min-plus framing: no negative-weight cycles), then $A^{*,N} = A^{*,n-1}$ for all $N \geq n-1$, and we write $A^* := A^{*,n-1}$. This is the full Kleene star. In min-plus, $A^*$ encodes all-pairs shortest paths: $(A^*)_{ij}$ = shortest-path length from $i$ to $j$.

**References for tropical background:** Cuninghame-Green 1979 (foundational), Butković 2010 (textbook), Maclagan-Sturmfels 2015 (geometry), Akian-Bapat-Gaubert 2007 (algebra handbook), Maragos-Charisopoulos-Theodosis 2021 ("Tropical Geometry and Machine Learning," Proc. IEEE — best ML-facing survey).

---

## 2. Background — what the reader needs in one page

We assume two things from the reader:

**(A) The original Linearizer construction.** Shocher-Berman-Hallak 2025 propose $f(x) = g_y^{-1}(A g_x(x))$ with $g_x, g_y$ invertible neural networks and $A$ a matrix. They show: induced vector-space operations $\oplus_g, \odot_g$ make this $f$ linear between two Hilbert spaces; composition is closed; SVD/pseudoinverse/idempotency transfer to $f$ via $A$. Applications: one-step flow matching, modular style transfer, idempotent generative networks.

**(B) Tropical algebra in deep learning.** Two threads:

*Descriptive:* Zhang-Naitzat-Lim 2018 showed ReLU networks compute tropical rational functions. Charisopoulos-Maragos 2018 extended to maxout. Brandenburg-Loho-Montúfar 2024 ("The Real Tropical Geometry of Neural Networks") gave a polyhedral parameter-space view. These works *analyze* standard networks through a tropical lens; they do not propose new tropical architectures.

*Constructive:* Morphological neural networks (Ritter-Sussner 1990s onward) use $(\max, +)$ as primitives. Recent work: Dimitriadis-Maragos 2025 ("Training Deep Morphological NNs as Universal Approximators"), Enaieh-Fercoq 2025 (sparse subgradient training), Yoshida et al. 2023 (tropical NNs with tropical-projective-torus inputs), TropNNC 2024 (compression). All operate in *standard* coordinates.

The literature gap we fill: nobody combines (i) a learned invertible coordinate map with (ii) exact tropical-linear core, yielding architectural enforcement of tropical-algebraic properties on the overall mapping $f$. See §11 for full prior-art differentiation.

---

## 3. Core theory

### 3.1 Definitions

**Definition 1 (Tropical Linearizer).** Given invertible neural networks $g_x: \mathcal{X} \to \mathbb{R}^N$, $g_y: \mathcal{Y} \to \mathbb{R}^M$, and a tropical matrix $A \in \mathbb{R}_{\max}^{M \times N}$, the tropical Linearizer is
$$f(x) := g_y^{-1}\bigl(A \otimes g_x(x)\bigr).$$

**Definition 2 (Induced tropical operations).** Given invertible $g: V \to \mathbb{R}^N$, define for $v_1, v_2 \in V$, $a \in \mathbb{R}$:
$$v_1 \oplus_g v_2 := g^{-1}\bigl(g(v_1) \oplus g(v_2)\bigr) = g^{-1}\bigl(\max(g(v_1), g(v_2))\bigr) \quad \text{(elementwise max)}$$
$$a \otimes_g v := g^{-1}\bigl(a \otimes g(v)\bigr) = g^{-1}\bigl(a + g(v)\bigr) \quad \text{(elementwise addition)}$$

Note: tropical scalar multiplication is real-number addition. This is the standard convention.

### 3.2 Lemmas

**Lemma 1 (induced tropical semimodule).** $(V, \oplus_g, \otimes_g)$ is a semimodule over $\mathbb{R}_{\max}$.

*Proof sketch.* Pull back the standard tropical semimodule structure on $\mathbb{R}^N$ through $g$. Semimodule axioms (associativity of $\oplus$, commutativity, idempotency $v \oplus_g v = v$, distributivity of $\otimes$ over $\oplus$, additive identity $g^{-1}(\mathbf{0})$, scalar identity) all transfer via the bijection. Note: tropical semimodules are *not* vector spaces — there are no additive inverses, since $\max$ has no inverse. ∎

**Lemma 2 (tropical linearity of $f$).** $f$ is tropical-linear: for $x_1, x_2 \in \mathcal{X}$, $a_1, a_2 \in \mathbb{R}$,
$$f\bigl(a_1 \otimes_x x_1 \oplus_x a_2 \otimes_x x_2\bigr) = a_1 \otimes_y f(x_1) \oplus_y a_2 \otimes_y f(x_2).$$

*Proof.* Expand:
\begin{align}
f(a_1 \otimes_x x_1 \oplus_x a_2 \otimes_x x_2) &= g_y^{-1}\bigl(A \otimes g_x(g_x^{-1}(\max(a_1 + g_x(x_1), a_2 + g_x(x_2))))\bigr) \\
&= g_y^{-1}\bigl(A \otimes \max(a_1 + g_x(x_1), a_2 + g_x(x_2))\bigr) \\
&= g_y^{-1}\bigl(\max(a_1 + A \otimes g_x(x_1), a_2 + A \otimes g_x(x_2))\bigr) \\
&= a_1 \otimes_y f(x_1) \oplus_y a_2 \otimes_y f(x_2). \quad \square
\end{align}
The third equality uses tropical linearity of $A$ on $\mathbb{R}^N$: $A \otimes (a + z) = a + A \otimes z$ and $A \otimes \max(u, v) = \max(A \otimes u, A \otimes v)$.

**Lemma 3 (composition closure).** Let $f_1: \mathcal{X} \to \mathcal{Y}$ and $f_2: \mathcal{Y} \to \mathcal{Z}$ be tropical Linearizers with compatible shared $g_y$. Then
$$(f_2 \circ f_1)(x) = g_z^{-1}\bigl((A_2 \otimes A_1) \otimes g_x(x)\bigr),$$
i.e., the composition is a tropical Linearizer with core $A_2 \otimes A_1$ (tropical matrix product).

*Proof.* Direct substitution, with the inner $g_y \circ g_y^{-1}$ canceling. ∎

**Lemma 4 (Kleene collapse — KEY result).** Let $f$ be a tropical Linearizer with shared $g_x = g_y = g$ and core $A \in \mathbb{R}_{\max}^{n \times n}$. Define the $N$-step accumulated trajectory operator
$$F_N(x) := \bigoplus_{k=0}^N f^{\circ k}(x) = x \oplus_g f(x) \oplus_g f(f(x)) \oplus_g \dots \oplus_g f^{\circ N}(x).$$
Then
$$F_N(x) = g^{-1}\bigl(A^{*,N} \otimes g(x)\bigr).$$
If $A$ has no positive-weight cycles, then $F_N(x) = F_{n-1}(x)$ for all $N \geq n-1$, and we write
$$F_\infty(x) = g^{-1}\bigl(A^* \otimes g(x)\bigr).$$

*Proof.* By induction and Lemma 3, $f^{\circ k}(x) = g^{-1}(A^{\otimes k} \otimes g(x))$. Then
$$F_N(x) = \bigoplus_{k=0}^N g^{-1}(A^{\otimes k} \otimes g(x)) = g^{-1}\left(\bigoplus_{k=0}^N A^{\otimes k} \otimes g(x)\right) = g^{-1}(A^{*,N} \otimes g(x)).$$
The middle step uses that $g^{-1}(\max(\cdot, \cdot))$ commutes with $\oplus_g$ by Def. 2. Stabilization follows from standard max-plus algebra: $A^{*,n-1}$ is the maximum-weight walk of length $\leq n-1$, and longer walks repeat cycles which don't increase weight under the no-positive-cycle assumption. ∎

**Interpretation.** Lemma 4 is the tropical analog of Eq. 17 in the original Linearizer paper (Euler iteration → matrix power). For shortest paths: $A$ encodes one Bellman-Ford update, $A^*$ encodes the all-pairs shortest-path table. *N iterations of a tropical Linearizer collapse into one application of the Kleene star.*

**Lemma 5 (tropical idempotency).** $f$ is tropically idempotent ($f \circ f = f$) iff $A \otimes A = A$.

*Proof.* $f \circ f = g^{-1}((A \otimes A) \otimes g(\cdot))$ by Lemma 3 with $g_x = g_y = g$. The equation $f \circ f = f$ is equivalent to $A \otimes A = A$ since $g^{-1}$ is a bijection. ∎

**Lemma 6 (Kleene stars are exactly the relevant tropical idempotents).** A zero-diagonal matrix $A \in \mathbb{R}_{\max}^{n \times n}$ satisfies $A \otimes A = A$ iff $A = B^*$ for some $B$, iff $A$ is a Kleene star in the sense of Puente 2013.

*Proof.* Standard result, see Puente 2013 "On tropical Kleene star matrices and alcoved polytopes" and Sergeev 2009. Briefly: a Kleene star $A^*$ satisfies $A^* \otimes A^* = A^*$ by definition. Conversely, if $C$ is zero-diagonal and tropically idempotent, then $C = C^*$. ∎

**Lemma 7 (column span as tropical polytope).** The image of $f$ (the "tropical column span" of $A$ transported by $g$) is the alcoved polytope $g^{-1}(\text{col}_{\max}(A))$, where $\text{col}_{\max}(A)$ is the max-plus convex hull of the columns of $A$. This is the tropical analog of the linear subspace image.

*Proof sketch.* The tropical column span is by definition $\{A \otimes z : z \in \mathbb{R}_{\max}^n\}$, which is a tropical (max-plus) polytope. Apply $g^{-1}$. ∎

**Lemma 8 (i-SVD of $f$ via i-SVD of $A$ — weak form).** If $A = U \otimes \Sigma \otimes V^*$ is an idempotent-semifield SVD of $A$ (Maragos-Theodosis), then transporting the singular vectors through $g^{-1}$ gives a corresponding decomposition of $f$.

*Caveat.* Unlike Euclidean SVD, i-SVD does not have unique orthogonal $U, V$ — the Galois connection gives two natural bases (join-dense vs. meet-dense). We state this lemma but do not use it for downstream applications. Included for completeness of the algebraic story.

### 3.3 What does NOT transfer from the linear case

Be explicit. These are NOT in the paper:

- **Pseudoinverse / inversion (Lemma 7 in the original paper).** Tropical Cuninghame-Green inverse requires $A$ to be regular (no zero columns/rows and certain rank conditions). We do not claim a tropical pseudoinverse application. No inversion/interpolation experiments.
- **Hilbert structure (Lemma 4 in the original paper).** Tropical semimodules are not inner-product spaces. We do not claim a tropical "inner product."
- **Transpose / adjoint (Lemma 5 in the original paper).** Tropical transpose exists ($A^T$ elementwise) but does not correspond to an adjoint w.r.t. an inner product.
- **Spectral theory.** Tropical eigenvalues exist (Cuninghame-Green eigenvalue = max cycle mean), but the spectrum is not the same object as Euclidean eigenvalues. We do not lean on this.

These omissions are honest and explicit — see §10 Limitations.

---

## 4. Architecture

### 4.1 The invertible map $g$

Use a standard normalizing flow architecture. Default choice: **RealNVP-style affine coupling layers** with invertible 1×1 convolutions (Glow-style), matching the original Linearizer paper's choices for direct comparability.

```
g = Sequential(
  [InvertibleBlock(channels=C, hidden=H) for _ in range(K)]
)

InvertibleBlock:
  ActNorm → Invertible 1×1 Conv → Affine Coupling
  (forward and inverse exact)
```

**Hyperparameters:**
- Number of blocks $K = 6$ (match original Linearizer).
- Hidden width $H$ determined by dataset (see §6.X).
- Image data: optional Squeeze2x2 to manage channel dimension.

**Important constraint.** $g$ is exactly invertible (Jacobian determinant tractable, but we don't need it — we are not modeling densities, only transporting algebraic structure). No softening, no approximate inverses.

### 4.2 The tropical core $A$

Three parameterization regimes depending on application:

**Regime A: Dense unconstrained tropical matrix.**
$$A_{ij} = \text{raw\_param}_{ij} \in \mathbb{R}, \quad i, j \in [n].$$
No $-\infty$ entries (or, if explicit sparsity is desired, learnable mask via Gumbel-sigmoid). Used for flow-matching-like applications, but NOT our primary setting.

**Regime B: Hypernetwork-produced $A$ conditioned on input.** For graph tasks where the graph structure determines the operator:
$$A = \text{Hyper}_\phi(\text{graph features}).$$
The hypernetwork is a small MLP or GNN that outputs the $n \times n$ tropical matrix. Used in CLRS Bellman-Ford and Warcraft setups.

**Regime C: Kleene-star-by-construction (for idempotent application).** Parameterize via raw $P \in \mathbb{R}^{n \times n}$, then
$$A := (I \oplus P)^{\otimes (n-1)}.$$
This is always a Kleene star (i.e., tropically idempotent) by construction, provided $P$ has non-positive diagonal (enforce $P_{ii} \leq 0$ via $-\text{softplus}$ of raw parameter, or zero by symmetry). Alternative: parameterize $A = P^*$ directly via the formula $A_{ij} = \max_{k \in [n-1]} (P^{\otimes k})_{ij}$, implemented via $n-1$ tropical matmuls in the forward pass.

For Regime C, the trade-off is: computing $A^*$ requires $O(n^3 \log n)$ operations per step (Floyd-Warshall-like). We include this cost in the training compute budget.

### 4.3 Tropical matrix-vector multiplication — PyTorch primitive

```python
def trop_matvec(A, x, infty=-1e9):
    """
    A: (m, n) tropical matrix
    x: (..., n) tropical vector
    returns: (..., m) tropical vector
    Computes (A ⊗ x)_i = max_j (A_{ij} + x_j)
    """
    # Broadcast: A[:, None, :] + x[..., None, :]
    # Shape: (..., m, n) after broadcast
    expanded = A + x.unsqueeze(-2)        # broadcast over batch dims
    return torch.max(expanded, dim=-1).values

def trop_matmul(A, B, infty=-1e9):
    """
    A: (m, k), B: (k, n)
    returns: (m, n) with (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})
    """
    # Use logsumexp trick for stability if soft-relaxed, else direct max
    return (A.unsqueeze(2) + B.unsqueeze(0)).max(dim=1).values
```

**Numerical note.** Use `-1e9` (not `-inf`) for the additive identity to avoid NaN propagation in autograd. Mask gradients on these positions during backprop if needed.

**Gradient flow.** `torch.max` gives sparse subgradients (1.0 at argmax, 0 elsewhere). This is the correct tropical subgradient. See §4.5 for handling sparsity in training.

### 4.4 Kleene-star computation — three implementations

For idempotent Regime C, three options to compute $A^*$:

**Option 1: Iterated tropical squaring.** $A^* = (I \oplus A)^{\otimes \lceil \log_2 n \rceil}$. Cost: $O(\log n)$ tropical matmuls = $O(n^3 \log n)$. Default choice.

**Option 2: Floyd-Warshall.** Standard $O(n^3)$ algorithm, but with `min` (or `max`) primitives that PyTorch can differentiate through. More memory-efficient than Option 1 for large $n$.

**Option 3: Power iteration to convergence.** Iterate $A^{k+1} = A^k \otimes A$ until $A^{k+1} = A^k$. Used for analysis, not training (no fixed step count).

Default: Option 1 for $n \leq 64$, Option 2 for larger.

### 4.5 Training — sparse subgradients

The tropical max gives subgradient = 1 at argmax, 0 elsewhere. This is sparse and causes slow training. Three mitigations, ordered by preference:

**Mitigation 1: Soft tropical relaxation with temperature annealing.** Replace `max` with `logsumexp / β`:
$$\text{softmax}_\beta(z) := \frac{1}{\beta} \log \sum_j \exp(\beta z_j).$$
As $\beta \to \infty$, this recovers exact tropical max. Anneal $\beta$ from a small value (e.g. 1.0) to large (e.g. 100) over training.

**Inference-time:** use exact $\max$. Soft form is training-only.

**Mitigation 2: Straight-through estimator.** Forward pass uses exact max; backward pass uses softmax with fixed temperature. Implement as:
```python
def st_max(z, beta):
    hard = torch.max(z, dim=-1).values
    soft = torch.logsumexp(beta * z, dim=-1) / beta
    return hard.detach() + (soft - soft.detach())
```

**Mitigation 3: Sparse subgradient optimizer.** Use the Enaieh-Fercoq 2025 sparse subgradient algorithm. Most effort, smallest expected gain; treat as a fallback.

**Decision:** Start with Mitigation 1 (annealed softmax). If training is unstable, fall back to Mitigation 2. Document the choice in the paper.

### 4.6 Reference architecture diagrams

```
                  ┌──────────┐         ┌──────────┐
                  │          │   A_⊗   │          │
   x  ──────────► │   g_x    │ ──────► │  g_y⁻¹   │ ──────► f(x)
                  │ (invert. │         │ (invert. │
                  │   flow)  │         │   flow)  │
                  └──────────┘         └──────────┘

   (in induced coords:  f is exactly tropical-linear, i.e. f(x) = A ⊗ x in latent)
```

For idempotent variant (Kleene star core):
```
   raw P  ─►  Project to "no positive cycles"  ─►  A* = (I ⊕ P)^⊗(n-1)
   
   Then: f(x) = g⁻¹(A* ⊗ g(x))  is tropically idempotent by construction.
```

---

## 5. Applications and experiments

Three applications, each tied to a standard benchmark suite. Each application section follows the same structure: **task**, **architecture instantiation**, **baselines**, **metrics**, **expected results**, **ablations**.

### 5.1 Application 1 — One-shot Bellman-Ford via Kleene collapse

**Task family:** Single-source and all-pairs shortest paths on weighted graphs, learned end-to-end from algorithm trajectories.

**Datasets:**
- **CLRS-30** (Veličković et al. 2022): Bellman-Ford, Dijkstra, Floyd-Warshall, BFS. Train on graphs of size $n=16$, test OOD on $n=64$. Standard splits.
- **ECHO benchmark** (Miglior-Tolloso-Gravina-Bacciu, Dec 2025, arXiv:2512.17762): single-source shortest paths, node eccentricity, graph diameter, on structurally challenging topologies. Recent and unsaturated.
- **Optional:** synthetic Erdős–Rényi and grid graphs for controlled scaling experiments.

**Architecture instantiation (Regime B + C hybrid).**
- Each node $i$ has a state $h_i^{(t)} \in \mathbb{R}^d$. Initial state: encode source indicator + node features.
- Tropical Linearizer update (replaces standard MPNN message passing):
$$h_i^{(t+1)} = g^{-1}\bigl(\text{trop-aggregate}_{j \in N(i) \cup \{i\}}\, A_{ij} \otimes g(h_j^{(t)})\bigr),$$
where $A_{ij} = \text{Hyper}_\phi(e_{ij})$ is produced from edge features $e_{ij}$ (which include edge weights) by a small MLP.
- "trop-aggregate" is the tropical $\bigoplus$ (elementwise max) over neighbors. This is the analog of `sum` in standard MPNNs.

**Key experimental claim — Kleene collapse.** Train the per-step tropical operator $A$, then evaluate two ways:
1. *Iterative:* run the tropical message-passing update $T$ times.
2. *One-shot:* compute the block-tropical Kleene star $\mathbf{A}^*$ over the node set, then $h^{(\infty)} = g^{-1}(\mathbf{A}^* \otimes g(h^{(0)}))$ in a single forward pass.

Show: one-shot and $T$-step results agree (up to numerical precision) when $T \geq n-1$. Show: OOD generalization (train $n=16$, test $n=64$) is *better* under one-shot than under fixed-$T$ MPNN baselines, because the one-shot Kleene operator handles arbitrary path lengths by construction.

**Baselines:**
- MPNN (standard message passing, sum aggregation).
- GAT (graph attention).
- PGN (Pointer Graph Networks, Veličković 2022) — the strongest CLRS baseline.
- Triplet-MPNN with hint supervision (CLRS-style).
- Iterative GNN with $T$ steps matched to $n-1$ (apples-to-apples).
- *Differentiable shortest-path solvers:* DataSP (Lahoud et al. 2024), Blackbox-Diff (Vlastelica/Pogančić 2020).

**Metrics:**
- CLRS: per-step output match accuracy, OOD accuracy at $n=64$.
- ECHO: MAE on shortest-path lengths, R² on eccentricity/diameter.
- One-shot vs. iterative agreement: max absolute deviation, mean relative error.

**Expected outcome.** Tropical Linearizer matches or beats MPNN/GAT in-distribution and shows meaningfully better OOD generalization (we hypothesize: +10–20 percentage points on CLRS Bellman-Ford at $n=64$) due to algorithmic alignment. We *will not* claim SOTA against heavily engineered methods like Triplet-MPNN with extensive hint supervision — the claim is about inductive bias, not engineering.

### 5.2 Application 2 — Image-to-path on Warcraft 12×12

**Task:** Given a 96×96 RGB Warcraft terrain image, predict the shortest path from top-left to bottom-right under a hidden cost embedding. This is the standard benchmark for differentiable combinatorial solvers (Vlastelica-Pogančić 2020, Petersen Newton Losses 2024).

**Architecture.**
- $g$: CNN-based invertible flow mapping $96 \times 96 \times 3$ to a flattened tropical latent space.
- $A$: hypernetwork producing the tropical cost matrix for the 12×12 grid graph from latent image features.
- Output: shortest path tensor (12×12 binary mask).

**Training loss.** Tropical cross-entropy on path edges, or direct MSE on the Kleene star output, depending on label format.

**Baselines (well-established):**
- Blackbox-Diff (Vlastelica-Pogančić 2020): the canonical baseline.
- Perturbed optimizers (Berthet et al. 2020).
- DataSP (Lahoud et al. 2024).
- Differentiable Dijkstra (DiffSortNet-derived).
- Newton Losses (Petersen et al. 2024).

**Metrics:**
- Path accuracy (exact match), cost accuracy, optimality gap.
- Standard Warcraft benchmark protocol; numbers directly comparable to published baselines.

**Expected outcome.** Competitive with state-of-the-art on standard Warcraft 12×12. Not the primary contribution — primary contribution is *that the architecture works* in a perception-to-combinatorial setting with no surrogate / smoothing, just exact tropical operations in learned coordinates.

### 5.3 Application 3 — Tropical Idempotent Generative Network (T-IGN)

**Task:** Train a tropically idempotent generator $f$ that projects any input onto a learned tropical polytope of the data distribution. Compare against the original IGN (Shocher et al. 2024).

**Architecture (Regime C).**
- Shared $g$.
- Core: $A = P^*$ computed via Kleene-star construction in §4.4.
- $f(x) = g^{-1}(P^* \otimes g(x))$ is tropically idempotent by construction (Lemma 5, 6).

**Loss.**
$$\mathcal{L} = \underbrace{\|f(x) - x\|_2^2}_{\text{reconstruction}} + \lambda_{\text{rank}} \cdot \text{tropical-rank}(P^*) + \lambda_{\text{iso}} \cdot \|\|g(x) - g(0)\|^2 - \|x\|^2\|_1.$$

The "tropical rank" regularizer encourages a low-dimensional tropical polytope. Use a relaxation: penalize the number of distinct columns of $P^*$ (cluster them via soft assignment) or use the Barvinok rank.

**Datasets:**
- MNIST 28×28 (apples-to-apples vs. IGN).
- CelebA 64×64 (apples-to-apples vs. IGN).
- *Piecewise-constant data* where the tropical-polytope inductive bias should genuinely help: binarized document images, mask datasets (e.g., COCO segmentation masks at low resolution), or simple line drawings (Quick-Draw). This is the *novel* setting; the tropical polytope structure is geometrically aligned with piecewise-constant manifolds.

**Baselines:**
- IGN (Shocher et al. 2024).
- Linear Linearizer IGN (Shocher-Berman-Hallak 2025 — directly comparable; same architecture family, Euclidean core).
- Approximate-idempotency methods (Jensen-Vicary 2025).
- Standard VAE / autoencoder (for reconstruction quality reference).

**Metrics:**
- Reconstruction (PSNR, LPIPS).
- Idempotency error: $\|f(f(x)) - f(x)\|_2$. Expected: numerically zero for T-IGN by construction; nonzero for IGN baselines.
- "Globality" of projection: project random noise inputs $z \sim \mathcal{N}(0, I)$ and measure whether projections land on the data manifold (FID against training set).
- Qualitative: visualize projections of OOD inputs (noise, other-class images) and observe piecewise-linear projection artifacts vs. Euclidean smoothness.

**Expected outcome.** T-IGN achieves exact idempotency (∼0 error) where baselines achieve ~10⁻² to 10⁻¹. On piecewise-constant datasets, T-IGN's projections should look qualitatively sharper / less blurred than Euclidean IGN, since the latter's linear span induces smoothing while the tropical polytope can have crisp piecewise-linear edges.

### 5.4 Cross-application: the composition collapse demonstration

This is the headline figure of the paper, analogous to the original Linearizer's Fig. 4(a).

**Experiment.** For Application 1 (CLRS Bellman-Ford):
- Train the tropical Linearizer to predict one Bellman-Ford step.
- Evaluate accuracy as a function of $T$ ∈ {1, 5, 10, 50, 100} iterative steps vs. one-shot Kleene-star inference.
- Plot accuracy vs. $T$; overlay one-shot Kleene as a horizontal line.

**Expected figure.** Iterative accuracy rises with $T$ and plateaus at $T = n-1$. One-shot Kleene matches the plateau exactly. The figure tells the story: *N iterations collapse into one Kleene application*, with no loss.

This is the empirical analog of the original paper's Eq. 17. Make it the central figure.

---

## 6. Implementation details

### 6.1 Repository structure (for Claude Code)

```
tropical-linearizer/
├── README.md                       # Project overview, install, run
├── pyproject.toml / requirements.txt
├── configs/                        # Hydra or YAML configs
│   ├── default.yaml
│   ├── app1_clrs.yaml
│   ├── app2_warcraft.yaml
│   └── app3_tign.yaml
├── tropical_linearizer/
│   ├── __init__.py
│   ├── tropical/
│   │   ├── ops.py                  # trop_matvec, trop_matmul, soft_max
│   │   ├── kleene.py               # Kleene star: squaring, Floyd-Warshall
│   │   └── core.py                 # TropicalCore module (Regime A/B/C)
│   ├── flows/
│   │   ├── coupling.py             # affine coupling layers
│   │   ├── glow.py                 # 1x1 conv, actnorm
│   │   └── invertible.py           # InvertibleFlow wrapper (forward/inverse)
│   ├── models/
│   │   ├── tropical_linearizer.py  # TropicalLinearizer = g_inv ∘ A_trop ∘ g
│   │   ├── tropical_mpnn.py        # GNN variant for graph tasks
│   │   └── tign.py                 # T-IGN variant with Kleene core
│   ├── data/
│   │   ├── clrs.py                 # CLRS-30 wrapper
│   │   ├── echo.py                 # ECHO benchmark loader
│   │   ├── warcraft.py             # Warcraft dataset
│   │   └── img.py                  # MNIST, CelebA, masks
│   ├── train/
│   │   ├── trainer.py              # generic training loop
│   │   ├── losses.py               # reconstruction, rank, isometry, etc.
│   │   ├── schedules.py            # β-annealing schedule for softmax
│   │   └── optim.py                # optimizer setup
│   ├── eval/
│   │   ├── metrics.py              # accuracy, PSNR, LPIPS, FID, idempotency
│   │   ├── ood.py                  # OOD evaluation harness for CLRS
│   │   └── collapse.py             # one-shot vs. iterative comparison
│   └── utils/
│       ├── logging.py              # WandB / TensorBoard
│       └── viz.py                  # plotting helpers
├── scripts/
│   ├── train_app1.py
│   ├── train_app2.py
│   ├── train_app3.py
│   └── reproduce_table_*.py
├── tests/
│   ├── test_tropical_ops.py        # validate trop matmul, Kleene
│   ├── test_idempotency.py         # numerically verify f(f(x)) = f(x)
│   ├── test_collapse.py            # N-step = one-shot Kleene
│   └── test_invertibility.py       # g(g^-1(x)) = x
└── paper/                          # LaTeX source (built in parallel)
    ├── main.tex
    ├── sections/
    └── figs/
```

### 6.2 Key implementation notes

**Tropical operations module (`tropical/ops.py`).** Implement both hard and soft variants. Soft variant takes `beta` as argument. Tests must verify:
- `trop_matmul(I, A) == A` (identity).
- `trop_matmul(A, B) @ x == trop_matmul(trop_matmul(A, B), x)` (associativity).
- Gradient through `torch.max` flows correctly.

**Kleene star (`tropical/kleene.py`).** Two implementations:
```python
def kleene_squaring(P, n_iter=None):
    """A* via iterated squaring: A* = (I ⊕ P)^⊗ceil(log2(n))."""
    n = P.shape[-1]
    n_iter = n_iter or math.ceil(math.log2(max(n, 2)))
    A = I_trop(n) + P  # I ⊕ P, but use tropical add (max)... actually:
    A = trop_add(I_trop(n), P)
    for _ in range(n_iter):
        A = trop_matmul(A, A)
    return A

def kleene_floyd_warshall(P):
    """Standard Floyd-Warshall, differentiable."""
    n = P.shape[-1]
    A = trop_add(I_trop(n), P)
    for k in range(n):
        # A_{ij} = max(A_{ij}, A_{ik} + A_{kj})
        A = torch.maximum(A, A[..., :, k:k+1] + A[..., k:k+1, :])
    return A
```

**Invertibility tests (`tests/test_invertibility.py`).** For every flow, verify on random inputs:
```python
x = torch.randn(B, C, H, W)
z, log_det = g.forward(x)
x_reconstructed = g.inverse(z)
assert torch.allclose(x, x_reconstructed, atol=1e-5)
```

**Idempotency tests (`tests/test_idempotency.py`).** For T-IGN:
```python
y = f(x)
yy = f(y)
assert torch.allclose(y, yy, atol=1e-4)
```

This MUST pass for the architecture to claim by-construction idempotency.

### 6.3 Hyperparameter defaults

| Parameter | Default | Notes |
|---|---|---|
| Flow blocks $K$ | 6 | Match original Linearizer |
| Hidden width $H$ | 64 (MNIST) / 128 (CelebA) | |
| Tropical latent dim $n$ | 64–256 | Trade off expressivity vs. Kleene cost ($O(n^3 \log n)$) |
| Softmax β schedule | 1 → 100 over 50k steps | Linear in log-space |
| Optimizer | AdamW, lr 1e-4, wd 1e-5 | |
| Batch size | 64 (img) / 32 (graph) | |
| Loss weights | $\lambda_\text{rank} = 0.5$, $\lambda_\text{iso} = 0.001$ | Same as IGN paper |
| Numerical "−∞" | $-10^9$ | |
| Training steps | 100k (Apps 1, 2), 200k (App 3) | |

### 6.4 Compute budget

Per-experiment estimates (single A100):
- Application 1 (CLRS, $n=16$): ~12 hours.
- Application 2 (Warcraft 12×12): ~24 hours.
- Application 3 (T-IGN, MNIST): ~12 hours. CelebA: ~48 hours.
- Total for full table reproduction: ~1 GPU-week per dataset family. Budget ~3 GPU-weeks total with ablations.

---

## 7. Ablations

### 7.1 Critical ablations (must include)

**A1. Identity vs. learned $g$.** Set $g = \text{Id}$ and train only $A$. This is "pure tropical neural network" (close to Yoshida 2023). Expected: significantly worse on tasks requiring perception (Warcraft) and OOD graph generalization. This isolates the contribution of *learned coordinates*.

**A2. Shallow vs. deep $g$.** Vary $K \in \{2, 4, 6, 8\}$ flow blocks. Expected: depth matters but plateaus around $K=6$.

**A3. Hard max vs. softmax-during-training.** Compare β-annealing schedule vs. constant high β vs. straight-through estimator. Expected: annealing wins; constant high β is unstable.

**A4. Kleene star degree.** For Application 1, train with truncated Kleene $A^{*,N}$ for $N \in \{1, 2, 4, 8, 16, n-1\}$. Show: $N = n-1$ is needed for full-graph reasoning; smaller $N$ degrades gracefully.

**A5. Linear core vs. tropical core.** Drop-in replacement: $A \otimes g(x)$ → $A \cdot g(x)$. This recovers the original Linearizer. Compare on graph tasks (Application 1). Expected: linear core fails at shortest-path tasks because composition is not algorithmically aligned.

**A6. Min-plus vs. max-plus.** Verify the two are equivalent under negation in all tasks. Sanity check.

### 7.2 Secondary ablations (nice-to-have)

**B1. Dense vs. low-rank tropical $A$.** Factorize $A = A_1 \otimes A_2$ with rank $r < n$. Trade-off between expressivity and compute.

**B2. Hypernetwork architecture for $A$.** MLP vs. small GNN vs. transformer for producing $A$ from graph features.

**B3. Effect of flow type.** RealNVP vs. Glow vs. neural spline flows.

**B4. Tropical rank regularizer strength.** Sweep $\lambda_\text{rank}$ in T-IGN.

---

## 8. Figures and tables for the paper

**Figures (proposed):**

1. **Fig. 1 — Architectural diagram.** Top: $f = g_y^{-1} \circ A_\otimes \circ g_x$. Bottom: induced tropical operations $\oplus_g$, $\otimes_g$. Direct analog of Linearizer Fig. 1.
2. **Fig. 2 — Composition collapse (HEADLINE).** Accuracy vs. $T$ iterative steps on CLRS Bellman-Ford, with one-shot Kleene overlay. Show plateau at $T = n-1$ matched by one-shot.
3. **Fig. 3 — OOD generalization on CLRS.** Bar chart: accuracy at $n = 16, 32, 48, 64$ for Tropical Linearizer vs. MPNN, GAT, PGN. Expect: our gap widens with $n$.
4. **Fig. 4 — Warcraft path predictions.** Qualitative: input image, predicted path, ground truth, for our method vs. Blackbox-Diff.
5. **Fig. 5 — Tropical polytope projections.** Visualize $f(x)$ for various inputs $x$ on a piecewise-constant dataset (binarized docs or masks). Compare T-IGN vs. linear IGN: T-IGN should show crisp piecewise-linear projection boundaries.
6. **Fig. 6 — Tropical polytope structure.** Visualize the learned column span of $A$ in 2D/3D projections of $g$-latent space. Show alcoved-polytope structure.

**Tables (proposed):**

1. **Tab. 1 — CLRS results.** All algorithms × all baselines, in-distribution and OOD accuracy.
2. **Tab. 2 — ECHO results.** MAE on SSSP, eccentricity, diameter; tropical Linearizer vs. baselines.
3. **Tab. 3 — Warcraft 12×12.** Path accuracy, cost accuracy, optimality gap.
4. **Tab. 4 — Idempotency error.** $\|f(f(x)) - f(x)\|$ for T-IGN vs. IGN vs. Jensen-Vicary, averaged over test set.
5. **Tab. 5 — Reconstruction quality (T-IGN).** PSNR/LPIPS on MNIST, CelebA, masks.
6. **Tab. 6 — Ablation summary.** A1–A6 from §7.1, compact.

---

## 9. Related work — precise differentiation

Cite, then differentiate, in this order:

**Original Linearizer (Shocher, Berman, Hallak, 2025; under review ICLR 2026).**
> "Our work directly extends the Linearizer framework from the linear semiring to the tropical semiring $(\mathbb{R}_{\max}, \max, +)$. We inherit the architectural template (invertible flow + algebraic core + invertible flow) and the composition-closure theorem, but transport a fundamentally different algebraic structure: max instead of plus for addition, plus instead of multiplication for scaling. This yields different applications (one-shot dynamic programming instead of one-shot ODE integration; tropical-polytope projection instead of linear-subspace projection)."

**Tropical Geometry of Neural Networks (Zhang-Naitzat-Lim 2018; Charisopoulos-Maragos 2018; Brandenburg-Loho-Montúfar 2024).**
> "These works *analyze* standard ReLU networks through a tropical lens, showing they realize tropical rational functions in Euclidean coordinates. We take a constructive stance: by inserting learned invertible coordinate maps before and after a tropical-linear core, we obtain neural networks that are *exactly* tropical-linear in induced coordinates by architectural design — not as a post hoc analytical observation."

**Tropical / Morphological Neural Networks (Ritter-Sussner; Charisopoulos-Maragos 2018; Smyrnis-Maragos 2020; Yoshida et al. 2023; Dimitriadis-Maragos 2025; Enaieh-Fercoq 2025).**
> "Tropical and morphological neural networks use max-plus as the primitive operation in standard coordinates, building expressive networks from tropical layers. Our distinction is two-fold: (i) the algebraic structure is transported *through* a learned diffeomorphism, so the network is tropical-linear in a learned latent space, not in input space; (ii) we obtain composition closure and Kleene-star collapse as architectural guarantees, which existing tropical NNs do not provide. The closest collision is Yoshida et al. 2023, which uses fixed tropical-projective-torus embeddings; we replace this fixed embedding with a learned invertible flow."

**Differentiable shortest-path solvers (Vlastelica-Pogančić 2020 Blackbox-Diff; Pogančić-Pailoor-Tarlow 2020; DataSP, Lahoud et al. 2024; Petersen Newton Losses 2024; DiffSortNets, Petersen et al. 2021).**
> "These methods smooth the discrete min/argmin to enable gradient flow through classical solvers. Our approach is structurally different: we do not smooth — we transport. The tropical operation is exact in latent coordinates; learning happens through the invertible coordinate map and the learned tropical operator $A$. This yields by-construction algorithmic alignment (one Bellman-Ford step = one tropical matvec) and natural composition collapse (N steps = one Kleene)."

**Neural Algorithmic Reasoning (Veličković-Blundell 2021; CLRS-30, Veličković 2022; Numeroso-Bacciu-Veličković 2023 Dual Algorithmic Reasoning).**
> "The NAR program aligns neural architectures with classical algorithms via inductive biases. Our contribution is a specific architectural realization: a tropical Linearizer's update rule *is* one Bellman-Ford step, by construction, in a learned latent space — and its composition collapses to one Kleene star. This is a tighter form of algorithmic alignment than message-passing approximations."

**Tropical matrix factorization (De Schutter-De Moor 2002; Karaev-Miettinen 2016 Capricorn/Cancer; Hook 2018; Karaev-Hook-Miettinen 2018 Latitude).**
> "Tropical matrix factorization decomposes a data matrix into tropical factors for interpretability. We use tropical matrices as the *core operator* in a learned neural pipeline, not as a factorization of fixed data; the matrix $A$ is learned end-to-end alongside the invertible coordinate maps."

**Normalizing flows (Rezende-Mohamed 2015; Dinh et al. 2015, 2017; Kingma-Dhariwal Glow 2018).**
> "We use normalizing-flow architectures as building blocks for $g$, but for transporting algebraic structure rather than probability measures. The Jacobian determinant is not used."

---

## 10. Limitations (be honest in the paper)

State plainly in §Limitations:

1. **Tropical pseudoinverse / inversion does not transfer.** The Cuninghame-Green inverse exists only under restrictive conditions; we do not claim a tropical pseudoinverse analog. No inversion or latent-interpolation experiments.

2. **Tropical SVD is weaker than Euclidean SVD.** The i-SVD (Maragos-Theodosis) provides a decomposition but lacks the uniqueness and orthogonality of Euclidean SVD. We state Lemma 8 for completeness but do not lean on it.

3. **Training is harder than the linear case.** Subgradient sparsity from `max` requires either softmax relaxation or specialized optimizers (Enaieh-Fercoq 2025). Convergence is slower and more sensitive to hyperparameters than the linear Linearizer.

4. **Kleene star is $O(n^3)$ or $O(n^3 \log n)$.** This bounds the tropical latent dimension $n$ we can use. Scaling to very high-dimensional latents (e.g., transformer-scale) would need approximation or structured tropical matrices.

5. **Expressivity is bounded by tropical-rational functions composed with diffeomorphisms.** Not all functions are representable. In particular, tropical-linear maps in latent are piecewise linear in input (since composition of piecewise linear and smooth is piecewise smooth-linear). For tasks requiring smooth nonlinear interpolation in input space, the Euclidean Linearizer may be preferable.

6. **Universal approximation status is open.** The Dimitriadis-Maragos 2025 result shows max-plus-min nets are universal approximators on compact domains; the corresponding result for $g^{-1} \circ A_\otimes \circ g$ is plausible but not proven. We do not claim a universal approximation theorem.

---

## 11. Risks and mitigations

**R1. Tropical training instability.** Highest-likelihood risk. Mitigation: anneal softmax β, use small learning rate, gradient clipping. Have Enaieh-Fercoq 2025 sparse-subgradient method as a backup.

**R2. Yoshida 2023 collision.** Reviewers may say "this is just learned-embedding tropical NN." Mitigation: ablation A1 (set $g = \text{Id}$) demonstrates the contribution of the *learned* invertible flow specifically. Frame the contribution explicitly.

**R3. Warcraft / CLRS results may not beat hand-tuned baselines.** Mitigation: position the contribution as *algorithmic alignment and by-construction guarantees* rather than SOTA. Strong story even at parity. If we beat OOD generalization, that's already enough.

**R4. T-IGN may not visibly improve on Euclidean IGN.** Mitigation: choose datasets where the tropical-polytope structure actually helps — piecewise-constant data (masks, line drawings). Skip CelebA if it doesn't help.

**R5. Reviewers ask about Hilbert structure / inner product.** Mitigation: address head-on in §Limitations. We are honest: tropical semimodules are not Hilbert spaces. The algebraic story is *composition closure + idempotency*, not the full Linearizer suite.

---

## 12. Writing plan

**Section structure (target 10 pages main + appendix):**

| Section | Pages | Notes |
|---|---|---|
| 1. Introduction | 1.0 | Lead with Kleene-collapse one-line pitch |
| 2. Background | 0.5 | Tropical semiring + Linearizer recap |
| 3. Tropical Linearizer framework | 2.0 | Definitions, Lemmas 1–7, intuition |
| 4. Architecture | 0.5 | Flow + tropical core + Kleene construction |
| 5. App 1: One-shot Bellman-Ford | 2.0 | CLRS + ECHO results, headline figure |
| 6. App 2: Warcraft path | 1.0 | Image-to-path, baselines table |
| 7. App 3: T-IGN | 1.5 | Idempotent projection, qualitative results |
| 8. Analysis & ablations | 1.0 | A1–A6 summary |
| 9. Related work | 0.5 | Differentiation in §9 |
| 10. Limitations & conclusion | 0.5 | Honest discussion |
| Appendix | — | Full proofs, Maragos-Theodosis i-SVD, additional experiments, implementation details |

**Writing order (recommended):**
1. Framework section (theory) — write first, fastest to lock down.
2. Application 1 (Bellman-Ford) — most novel result, write next.
3. Headline composition-collapse figure (Fig. 2) — design and produce early; this is the paper's identity.
4. Ablations table — write alongside experiments.
5. Application 3 (T-IGN) — write after Application 1.
6. Application 2 (Warcraft) — write last; treat as confirmation rather than headline.
7. Related work and Limitations — final pass with full results in hand.
8. Introduction — write last, top-down from results.
9. Abstract — final.

---

## 13. Collaborators and division of labor

**Lead:** Assaf Hallak. Theory + Application 1 + headline figure.

**Suggested ask list (TBD, route through NVIDIA channels per IP policy):**
- **Assaf Shocher** (original Linearizer first author): co-author candidate. Owns the original framework; tropical extension naturally fits. Confirm IP/scope alignment with NVIDIA.
- **Nimrod Berman** (KDM, Linearizer collaborator): could own App 2 (Warcraft + perception pipeline).
- **Maragos lab (Charisopoulos, Smyrnis, Theodosis)** or **Yoshida group**: optional external sanity check on tropical algebra, NOT co-authors unless they actively contribute. Reach out for advice on tropical training stability.
- **NVIDIA team** (Eli Meirom, Yftah Ziser, Chen Tessler): collaborator candidates depending on bandwidth. Application 3 (T-IGN on piecewise-constant data) is a natural fit for Eli's geometric learning interests.

**Open question:** does this paper fit the NVIDIA RL-adjacent research agenda you committed to last quarter? Tropical Linearizer is closer to representation learning / structured architectures than to RL. Decision required before kickoff: is this a 20% project, a parallel thread, or your primary push for ICLR 2027? See §15.

---

## 14. Reproducibility checklist

- [ ] All code released under Apache 2.0 on GitHub.
- [ ] Pretrained checkpoints for all reported numbers.
- [ ] Hydra/YAML configs for every experiment in the paper.
- [ ] `scripts/reproduce_table_N.py` for each table.
- [ ] Unit tests verifying tropical ops, invertibility, idempotency, Kleene collapse.
- [ ] WandB project link or equivalent for training curves.
- [ ] Compute and runtime estimates in README.
- [ ] CLRS, ECHO, Warcraft datasets sourced from official repos; no modifications.

---

## 15. Decision points for Assaf (resolve before kickoff)

1. **Scope commitment.** Primary push for ICLR 2027 or parallel thread? Estimate ~3 GPU-weeks + 4–6 weeks of focused work.

2. **Co-authors.** Shocher in or out? Decision affects framing (sequel paper vs. independent extension) and IP routing.

3. **Application priority.** If forced to drop one, which? Recommendation: keep App 1 (CLRS) as primary, App 3 (T-IGN) as secondary, App 2 (Warcraft) as confirmation. App 2 is the most replaceable.

4. **Theoretical depth.** Include i-SVD lemma (Lemma 8) in main paper or appendix? Recommendation: appendix unless reviewer feedback demands otherwise.

5. **Soft tropical vs. hard tropical at inference.** Default to hard max at inference (exact tropical operation). Confirm OK with reviewers if soft variant is needed for downstream gradient pipelines — probably not relevant since downstream uses are out of scope.

6. **Universal approximation claim.** Pursue a UA theorem? Recommendation: no for the empirical paper. Note as future work. If pursued, may be a separate theory paper (potential thread with Nadav).

7. **Tropical i-SVD application.** Do we showcase Lemma 8 with an experiment, or just state it? Recommendation: state only. Adding an application would dilute the focus.

---

## 16. Bibliography seeds (for `paper/refs.bib`)

Core (Linearizer + this work):
- Shocher, Berman, Hallak. "Who Said Neural Networks Aren't Linear?" ICLR 2026 (under review).
- This paper.

Tropical algebra foundations:
- Cuninghame-Green. *Minimax Algebra*. Springer, 1979.
- Butković. *Max-Linear Systems: Theory and Algorithms*. Springer, 2010.
- Maclagan, Sturmfels. *Introduction to Tropical Geometry*. AMS, 2015.
- Akian, Bapat, Gaubert. "Max-plus algebra." *Handbook of Linear Algebra*, 2007.

Tropical / morphological in deep learning:
- Zhang, Naitzat, Lim. "Tropical Geometry of Deep Neural Networks." ICML 2018.
- Charisopoulos, Maragos. "A Tropical Approach to Neural Networks with Piecewise Linear Activations." 2018.
- Maragos, Charisopoulos, Theodosis. "Tropical Geometry and Machine Learning." *Proc. IEEE*, 2021.
- Brandenburg, Loho, Montúfar. "The Real Tropical Geometry of Neural Networks." 2024.
- Yoshida et al. "Tropical Neural Networks." 2023.
- Dimitriadis, Maragos. "Training Deep Morphological NNs as Universal Approximators." 2025.
- Enaieh, Fercoq. "Exploiting Subgradient Sparsity in Max-Plus Neural Networks." 2025.
- Smyrnis, Maragos. "Tropical Polynomial Division and Neural Networks." 2019.
- Maragos, Theodosis. "Singular Value Decomposition over Completed Idempotent Semifields." 2020.

Tropical Kleene / idempotent matrices:
- Puente. "On tropical Kleene star matrices and alcoved polytopes." *Kybernetika*, 2013.
- Sergeev. "Multiorder, Kleene stars and cyclic projectors in the geometry of max cones." 2009.

Differentiable solvers and sorting:
- Vlastelica, Pogančić, et al. "Differentiation of Blackbox Combinatorial Solvers." ICLR 2020.
- Pogančić, et al. "Learning with Combinatorial Optimization Layers." ICLR 2020.
- Lahoud, Schaffernicht, Stork. "DataSP." 2024.
- Petersen et al. "Differentiable Sorting Networks." 2021.
- Petersen et al. "Newton Losses." 2024.

Neural algorithmic reasoning:
- Veličković, Blundell. "Neural Algorithmic Reasoning." 2021.
- Veličković et al. "The CLRS Algorithmic Reasoning Benchmark." ICML 2022.
- Miglior et al. "ECHO: A Benchmark for Long-Range Graph Propagation." 2025.

Invertible networks / flows:
- Dinh et al. "NICE." ICLR-W 2015.
- Dinh et al. "RealNVP." ICLR 2017.
- Kingma, Dhariwal. "Glow." NeurIPS 2018.
- Rezende, Mohamed. "Normalizing Flows." ICML 2015.

Tropical matrix factorization:
- De Schutter, De Moor. "The QR decomposition and the singular value decomposition in the symmetrized max-plus algebra." 1998.
- Karaev, Miettinen. "Cancer: Another Algorithm for Subtropical Matrix Factorization." ECML 2016.
- Karaev, Miettinen. "Algorithms for Approximate Subtropical Matrix Factorization." DAMI 2018.
- Hook. "Linear regression over the max-plus semiring." 2018.
- Tsiamyrtzis et al. "Matrix Factorization in Tropical and Mixed Tropical-Linear Algebras." 2023.

IGN:
- Shocher et al. "Idempotent Generative Network." ICLR 2024.
- Jensen, Vicary. "Enforcing Idempotency in Neural Networks." ICML 2025.
- Durasov et al. "IT3: Idempotent Test-Time Training." ICML 2025.

---

## 17. Implementation milestones (4–6 week plan)

**Week 1: Foundations.**
- Repo scaffold, tropical ops with tests, invertible flow with tests.
- Kleene star implementations + numerical validation against NumPy / NetworkX.
- Run sanity check: tropical Linearizer with identity $g$ matches direct Floyd-Warshall on a small graph.

**Week 2: Application 1 prototype.**
- CLRS-30 data pipeline.
- Train tropical Linearizer on Bellman-Ford with $n=8$. Verify Kleene collapse experiment passes.
- First version of headline Fig. 2.

**Week 3: Application 1 scale-up + ablations.**
- Scale to $n=16$ training, $n=32, 64$ OOD eval.
- Run all critical ablations A1–A6.
- ECHO benchmark added.

**Week 4: Application 2.**
- Warcraft 12×12 pipeline.
- CNN-flow architecture for $g$.
- Baseline comparison run.

**Week 5: Application 3.**
- T-IGN with Kleene-star core.
- Verify idempotency to numerical precision.
- Run on MNIST + piecewise-constant dataset.

**Week 6: Polish.**
- Final ablations, qualitative figures, paper writing.
- Pre-submission internal review.

---

## 18. Quick gut check — should we actually do this?

**Reasons yes:**
- Architectural delta is clean and confirmed novel.
- Three benchmarks with established baselines = empirical paper writes itself.
- Headline Fig. 2 (Kleene collapse) is a clear, intuitive, memorable result.
- Strong narrative: original Linearizer told the linear-algebra-in-disguise story; this tells the tropical-algebra-in-disguise story, with shortest paths as the killer demo.
- Direct extension of your prior work — low ramp-up cost.

**Reasons to slow down:**
- Tropical training is genuinely harder than linear. Plan extra time for stabilization.
- Yoshida 2023 collision is real; differentiation must be clean from the start.
- Not RL-adjacent. If your NVIDIA mandate is strictly RL/LLM, this needs sponsorship discussion before commitment.
- App 3 (T-IGN) may not produce visually striking results on standard datasets; choose the comparison dataset carefully.

**Recommendation:** This is a strong paper concept. Resolve §15 decision points first (especially scope and co-authors), then commit. If RL/LLM mandate is strict, this is a parallel project at 30% time, not a primary push.

---

*End of blueprint. Hand off to Claude Code with the prompt: "Implement the Tropical Linearizer following this blueprint. Start with Week 1 milestones in §17."*
