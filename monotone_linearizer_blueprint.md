# Monotone Linearizer — Paper & Implementation Blueprint

**Working title:** *Monotone Linearizers: Expressivity-Preserving Equilibrium Models via Learned Coordinates*

**Target venue:** ICLR 2027 (primary). NeurIPS 2026 if timeline permits and execution is fast.

**Type:** Empirical paper with a clean theoretical foundation. Extends the Linearizer framework (Shocher-Berman-Hallak, ICLR 2026 under review) by replacing the linear semiring core with a monotone-operator core, targeting the well-known expressivity-stability tradeoff in implicit deep learning.

**Owner / lead:** Assaf Hallak. Co-author candidates discussed in §13.

**Status:** Blueprint v0. Hand off to Claude Code for both writing scaffold and implementation.

---

## 0. Quick orientation — the one-paragraph pitch

**The wish.** "If only this deep equilibrium model had guaranteed unique fixed points and stable convergence."

**The problem.** Standard DEQs (Bai-Koltun-Kolter 2019) are expressive but unstable: they diverge, have multiple equilibria, and require heavy regularization. Monotone DEQs (monDEQ, Winston-Kolter 2020) fix this by constraining the layer operator to be monotone — but the constraint $I - W \succeq mI$ is enforced in raw input coordinates, where most useful operators aren't naturally monotone. The literature openly acknowledges this expressivity cost.

**Our contribution.** We learn an invertible coordinate map $g$ such that the equilibrium operator is monotone in *induced* coordinates. The monotonicity guarantee (and thus uniqueness, stability, and convergence rate) is preserved in the induced inner product, while the diffeomorphism $g$ recovers expressivity. Formally:
$$f(z, x) = g^{-1}\bigl(M(g(z), \varphi(x))\bigr)$$
where $M$ is a monDEQ-style monotone operator in latent and $g$ is a normalizing flow. The fixed point $z^* = f(z^*, x)$ exists, is unique, and is computed by solving the monotone fixed-point equation $w^* = M(w^*, \varphi(x))$ in latent (guaranteed convergent via forward-backward or Peaceman-Rachford splitting), then setting $z^* = g^{-1}(w^*)$.

**Why a Linearizer.** This is not just "wrap monDEQ in a flow" — the Linearizer framework gives us composition closure, a clean theoretical story via induced inner products, and explicit characterization of when the architecture buys something over the alternatives. We make the "non-trivial $g$ earns its keep" claim falsifiable: ablation A1 (set $g = \text{Id}$) recovers monDEQ exactly, and we show this ablation underperforms on tasks where standard DEQ has been competitive.

**The headline result.** On established DEQ benchmarks (CIFAR-10/100 classification, WikiText-103 language modeling, OGB long-range graph tasks), the Monotone Linearizer matches or exceeds standard DEQ accuracy *while retaining monDEQ's convergence guarantees* (unique equilibria, provable stability, bounded iteration count). This is a clean Pareto improvement over both endpoints.

---

## 1. Notation and preliminaries

**Monotone operator.** $F: \mathbb{R}^n \to \mathbb{R}^n$ is monotone if $\langle F(u) - F(v), u - v \rangle \geq 0$ for all $u, v$. It is $m$-strongly monotone if the inequality is $\geq m \|u - v\|^2$. References: Bauschke-Combettes 2017 (canonical textbook), Ryu-Boyd 2016 (operator splitting tutorial).

**Maximally monotone.** Cannot be extended to a larger monotone operator. Sufficient for the existence of resolvents and proximal operators.

**Resolvent.** $J_{\lambda F} := (I + \lambda F)^{-1}$. If $F$ is maximally monotone, $J_{\lambda F}$ is single-valued, firmly non-expansive, and defined on all of $\mathbb{R}^n$.

**Forward-backward splitting.** To find $z^*$ solving $0 \in (A + B)(z^*)$ for $A$ maximally monotone and $B$ cocoercive: iterate $z^{k+1} = J_{\lambda A}(z^k - \lambda B(z^k))$. Converges under cocoercivity + appropriate $\lambda$.

**Induced inner product (from original Linearizer).** Given invertible $g: V \to \mathbb{R}^n$:
$$\langle u, v \rangle_g := \langle g(u), g(v) \rangle.$$

**Monotonicity in induced metric.** $F: V \to V$ is *$g$-monotone* if $\langle F(u) - F(v), u - v \rangle_g \geq 0$ for all $u, v \in V$. Equivalently: $g \circ F \circ g^{-1}$ is monotone in standard Euclidean inner product on $\mathbb{R}^n$.

---

## 2. Background — what the reader needs in one page

**(A) Deep Equilibrium Models (DEQs).** Bai-Koltun-Kolter NeurIPS 2019 introduced implicit models defined by fixed points: $z^* = f_\theta(z^*, x)$. Memory-efficient (constant-memory backprop via implicit function theorem), competitive with deep transformers on LM and vision tasks. *Pain points:* no guarantee of fixed-point existence or uniqueness; iterations often diverge or oscillate; training requires regularization (Jacobian penalty, Bai-Koltun-Kolter ICML 2021).

**(B) Monotone Operator DEQs (monDEQ).** Winston-Kolter NeurIPS 2020. Constrain the operator so $z - f_\theta(z, x)$ is $m$-strongly monotone in $z$, via spectral constraint $I - W \succeq mI$. Result: unique equilibrium exists, forward-backward and Peaceman-Rachford splitting converge with rate guarantees. *Pain point:* the spectral constraint restricts expressivity. Acknowledged in the paper itself and addressed (partially) by follow-ups (Baker-Wang-Hauck-Wang ICML 2023 for IGNNs; Pabbaraju-Winston-Kolter ICLR 2021 on Lipschitz constants).

**(C) The Linearizer framework.** Shocher-Berman-Hallak 2025 (ICLR 2026 under review). $f(x) = g_y^{-1}(A g_x(x))$ with $A$ linear and $g_x, g_y$ invertible. Composition closure, SVD, idempotency transfer from $A$ to $f$ via induced operations. Applications: one-step flow matching, modular style transfer, idempotent generative networks.

**Our claim.** Replacing the linear core $A$ with a monotone operator $M$ extends the Linearizer framework to implicit models. The induced operations (inner product, monotonicity) transfer cleanly. The composition story holds with appropriate cocoercivity assumptions. And — critically — the resulting architecture lifts monDEQ's expressivity restriction by transporting the monotonicity property into a learned coordinate system, rather than enforcing it in raw input coordinates.

---

## 3. Core theory

### 3.1 Definitions

**Definition 1 (Monotone Linearizer Layer).** Given:
- invertible $g: V \to \mathbb{R}^n$ (normalizing flow),
- an $m$-strongly monotone operator $M: \mathbb{R}^n \times \mathbb{R}^d \to \mathbb{R}^n$ in its first argument,
- input encoder $\varphi: \mathcal{X} \to \mathbb{R}^d$,

define the Monotone Linearizer layer
$$f(z, x) := g^{-1}\bigl(M(g(z), \varphi(x))\bigr).$$
Its equilibrium-defining operator is $G(z, x) := z - f(z, x)$.

**Definition 2 (Monotone Linearizer DEQ).** The model output is $z^* = z^*(x)$ solving the fixed-point equation
$$z^* = f(z^*, x).$$
Equivalently in latent: $w^* = M(w^*, \varphi(x))$ with $w^* = g(z^*)$.

Notation: $\Phi(x) := z^*(x)$ is the model's input-to-output map.

### 3.2 Lemmas

**Lemma 1 (existence and uniqueness of $z^*$).** Under Definition 1's assumptions, the latent fixed-point equation $w^* = M(w^*, \varphi(x))$ has a unique solution $w^*(x)$, and consequently $z^*(x) = g^{-1}(w^*(x))$ is uniquely defined for every $x \in \mathcal{X}$.

*Proof.* By Winston-Kolter Theorem 1 (NeurIPS 2020), $m$-strong monotonicity of $w \mapsto w - M(w, \varphi(x))$ implies existence and uniqueness of $w^*$. The bijection $g$ transfers this to $z^*$. ∎

**Lemma 2 ($g$-monotonicity of the equilibrium operator).** $G(z, x) = z - f(z, x)$ is $m$-strongly monotone in $z$ with respect to the induced inner product $\langle \cdot, \cdot \rangle_g$:
$$\langle G(u, x) - G(v, x), u - v \rangle_g \geq m \|u - v\|_g^2.$$

*Proof.* Compute:
\begin{align}
\langle G(u, x) - G(v, x), u - v \rangle_g &= \langle g(G(u,x)) - g(G(v,x)), g(u) - g(v) \rangle.
\end{align}
Now, $g(G(u, x)) = g(u - f(u, x)) = g(u) - g(f(u, x)) + r$ where $r$ is a nonlinear residual. This step requires care — $g$ is not linear so $g(u - f(u, x)) \neq g(u) - g(f(u, x))$ in general. **The clean formulation uses induced operations: $G(u, x) = u \ominus_g f(u, x)$ where $\ominus_g$ is induced subtraction.** With that, the proof works cleanly. See full proof in Appendix.

*Operational meaning.* The standard convergence guarantees of forward-backward and Peaceman-Rachford splitting hold for the iteration on $w$ in latent coordinates. Equivalently, they hold in $z$ coordinates under the $g$-induced metric.

**Critical clarification.** The cleanest statement and the one we'll use throughout: *the latent equation $w^* = M(w^*, \varphi(x))$ is solved by standard monDEQ machinery in standard Euclidean coordinates on $\mathbb{R}^n$, and the output is obtained as $z^* = g^{-1}(w^*)$.* This avoids the technicality of induced operations on the residual $G$ and gives an honest algorithm.

**Lemma 3 (composition closure).** Let $f_1, f_2$ be Monotone Linearizer layers with shared latent dimension $n$ and *separate* invertible maps $g_1, g_2$ and cores $M_1, M_2$ where $M_1, M_2$ are both $m$-strongly monotone. Then the composition $f_2 \circ f_1$ is a Monotone Linearizer layer with:
- invertible map $g_1$ at the input side and $g_2$ at the output side (or shared if $g_1 = g_2$),
- core $M_{\text{comp}}(\cdot, \varphi(x)) = M_2(g_2(g_1^{-1}(M_1(\cdot, \varphi(x)))), \varphi(x))$.

*Caveat.* The composed core is monotone only under additional cocoercivity assumptions; monotone-on-monotone composition does not automatically yield monotone. This is a real limitation versus the linear Linearizer (where composition is automatic). See §10.

**Lemma 4 (computational equivalence to monDEQ in latent).** Under Definitions 1-2, computing $\Phi(x)$ is equivalent to:
1. Encode $\varphi(x)$.
2. Solve $w^* = M(w^*, \varphi(x))$ in $\mathbb{R}^n$ via forward-backward splitting: $w^{k+1} = J_{\lambda(I - M(\cdot, \varphi(x)))}(w^k)$. Converges in $O(\log(1/\epsilon))$ iterations.
3. Output $z^* = g^{-1}(w^*)$.

*Implication.* The forward pass uses standard monDEQ machinery in latent. The flow $g$ adds two extra costs: forward $g$ application (for input transformation if needed; in our default architecture only $g^{-1}$ is applied) and a single $g^{-1}$ at the end. Both are $O(\text{flow cost})$ per inference, which is small relative to the $O(K \cdot n^2)$ cost of $K$ DEQ iterations.

**Lemma 5 (implicit differentiation through $\Phi$).** $\nabla_x \Phi(x)$ is computed via implicit function theorem applied to $w^* = M(w^*, \varphi(x))$:
$$\nabla_x w^* = (I - \partial_w M)^{-1} \, \partial_\varphi M \cdot \nabla_x \varphi(x).$$
The Jacobian $(I - \partial_w M)$ is invertible by $m$-strong monotonicity. Then $\nabla_x \Phi(x) = \nabla g^{-1}(w^*) \cdot \nabla_x w^*$. The flow Jacobian $\nabla g^{-1}$ is tractable.

*Implication.* Backpropagation cost is one linear solve plus one flow Jacobian, same as monDEQ plus a flow-Jacobian term.

### 3.3 The "non-trivial $g$ earns its keep" theorem (informal claim, formal version TBD)

**Claim.** There exist input-to-output maps $\Phi: \mathcal{X} \to \mathcal{Z}$ realizable by a Monotone Linearizer DEQ with non-trivial $g$ but not realizable by any monDEQ ($g = \text{Id}$) of bounded width and depth.

*Intuition.* monDEQ's equilibrium operator is constrained to $z = z - \sigma(Wz + Ux + b)$ with $W$ in a specific spectral set. The implicit function $x \mapsto z^*(x)$ is therefore a particular family. Adding a flow $g$ after the equilibrium allows arbitrary diffeomorphic post-processing of this family, strictly enlarging the function class.

*Formal version (TODO).* Construct an explicit $\Phi$ — e.g., a learned input-to-output map where $z^*$ in standard coordinates does not satisfy monDEQ's spectral structure but does in latent — and show monDEQ cannot represent it. The proof goes through dimension-counting / parameter-counting arguments. Defer to the paper.

*Empirical version.* Ablation A1: set $g = \text{Id}$ and show CIFAR-100 / WikiText / OGB benchmarks underperform Monotone Linearizer DEQ by $\geq$ 3% accuracy / $\geq$ 5 perplexity. This is the empirical proxy for the theorem.

### 3.4 What does NOT transfer from the linear Linearizer

Be explicit. The linear Linearizer's Lemmas 4-8 (Hilbert structure, transpose, SVD, pseudoinverse) do NOT all transfer:

- **SVD.** Monotone operators don't admit SVD in the same way. Skip.
- **Pseudoinverse.** Resolvents replace pseudoinverses: $J_{\lambda M}$ is the natural "inverse-like" object for monotone $M$. We can state $J_{\lambda f}(y) = g^{-1}(J_{\lambda M}(g(y)))$ — the resolvent of $f$ in induced metric. This is a real result and we include it.
- **Transpose / adjoint.** Monotone operators have a notion of adjoint via subdifferentials. Skip for the empirical paper.
- **Hilbert structure.** The induced inner product gives a Hilbert space, but the operational use is captured by Lemma 4 (latent computation in Euclidean), so we mostly avoid Hilbert-space machinery in the main text. Move to appendix.

---

## 4. Architecture

### 4.1 Top-level model

```
Input x
  │
  ▼
[Encoder φ]                          (any standard architecture: CNN, MLP, GNN)
  │
  ▼ φ(x) ∈ ℝ^d
  │
  │  ┌────────────────────────────────────────┐
  │  │ Latent fixed-point solver (monotone)   │
  │  │                                        │
  └──┤   w* = M(w*, φ(x))                     │   (P1: monDEQ-style)
     │   via forward-backward splitting       │
     │                                        │
     └──────────────┬─────────────────────────┘
                    │ w* ∈ ℝ^n
                    ▼
                [Flow g⁻¹]                     (normalizing flow, e.g. RealNVP)
                    │
                    ▼ z* ∈ ℝ^n
                    │
                [Head]                         (e.g., classification logits, LM logits)
                    │
                    ▼
                Output
```

Note: the input $x$ does NOT pass through $g$. Only the *state* $z$ does. The flow $g$ transports the equilibrium solution from "monotone-friendly" latent coordinates to a richer state space.

### 4.2 The monotone core $M$

Three parameterizations, ordered by recommendation:

**P1 (default): monDEQ-style spectral constraint.**
$$M(w, c) = \sigma(W w + U c + b),\quad I - W \succeq m I,\quad m > 0.$$
Parameterize $W = (1-m) I - A^T A$ where $A$ is unconstrained. Then $I - W = m I + A^T A \succeq m I$ by construction.

Activation $\sigma$: ReLU is the canonical choice. The monDEQ paper proves $m$-strong monotonicity of $z \mapsto z - \sigma(Wz + \cdot)$ when $I - W \succeq mI$ and $\sigma$ is 1-Lipschitz monotone (which ReLU satisfies).

**P2: Baker-Wang-Hauck-Wang 2023 IGNN parameterization.** Less restrictive spectral set preserving well-posedness. Use for graph-structured tasks where the original monDEQ constraint is too tight.

**P3: Convex potential.** $M(w, c) = \nabla_w \Psi(w, c)$ where $\Psi$ is convex in $w$, parameterized via ICNN (Amos 2017). Monotonicity follows from convexity (gradient of convex = monotone). Higher computational cost but richer function class.

Default for all experiments: P1.

### 4.3 The invertible flow $g$

Standard choices. Default: **Glow-style with affine coupling + invertible 1×1 conv + ActNorm.** Number of blocks $K = 6$ (match the original Linearizer paper for direct comparability).

For different data types:
- **Images:** Glow with multi-scale Squeeze2x2 structure.
- **Sequences (LM):** stack of invertible 1×1 conv + coupling, treating sequence positions as channels.
- **Graphs:** node-level coupling layers (or treat node embeddings as a tensor and apply image-style flow).

**Critical constraint.** $g$ must be exactly invertible. No softening, no approximate inverse. Forward and inverse are explicit.

### 4.4 The fixed-point solver

Use the standard monDEQ solvers:
- **Forward pass:** Peaceman-Rachford splitting on $w = M(w, c)$. Converges in $O(\log(1/\epsilon))$ iterations with rate determined by $m$ and Lipschitz constant of $M$.
- **Backward pass:** implicit differentiation via one linear solve, $(I - \partial_w M)^{-1} v$. Use GMRES or Broyden's method as in Bai-Koltun-Kolter 2019.

### 4.5 Training stability

monDEQ training is more stable than vanilla DEQ but still requires:
- Jacobian regularization (Bai-Koltun-Kolter ICML 2021): small penalty on $\|\partial_w M\|$ to keep iterations well-conditioned.
- Equilibrium tolerance scheduling: start with loose tolerance, tighten during training.
- Gradient clipping at norm 1.0.

Adding the flow $g$ does not affect monDEQ's stability properties since $g$ doesn't appear in the latent fixed-point loop. Flow training is straightforward (no log-det Jacobian needed since we're not modeling densities).

---

## 5. Applications and experiments

Three benchmark suites, each chosen to directly compete with monDEQ and standard DEQ on their established turf.

### 5.1 Application 1 — Image classification (CIFAR-10, CIFAR-100)

**Task:** Standard supervised image classification, head on top of the equilibrium $z^*$.

**Why this benchmark.** monDEQ originally reported on CIFAR-10 (76.4% accuracy with a specific architecture). Standard DEQ achieves ~88% with more capacity. The gap is exactly the expressivity restriction we're attacking.

**Architecture.**
- Encoder $\varphi$: small CNN (3 conv layers, 64-128 channels).
- Latent dim $n = 256$.
- Monotone core: P1 with $m = 0.1$, ReLU activation.
- Flow $g$: 6 affine coupling blocks with 1×1 conv permutations.
- Head: linear classifier on $z^*$.

**Baselines.**
- monDEQ (Winston-Kolter 2020): direct apples-to-apples.
- Standard DEQ (Bai et al. 2019): expressive but unstable.
- ResNet-18, ResNet-50: explicit-depth references.
- Jacobian-regularized DEQ (Bai-Koltun-Kolter 2021).
- RevDEQ (Sept 2025): recent reversible DEQ variant.

**Metrics.**
- Top-1 / Top-5 accuracy.
- Equilibrium error: $\|w^* - M(w^*, \varphi(x))\|$ at convergence.
- Forward iterations to convergence (FLOPS proxy).
- Training stability: variance of training loss across seeds (3 seeds).

**Expected outcome.** Match or beat standard DEQ accuracy (within 1-2% of best DEQ result on CIFAR-100). Match monDEQ's stability (zero training divergences across seeds). This is the Pareto-improvement story — the headline of Application 1.

### 5.2 Application 2 — Language modeling (WikiText-103)

**Task:** Word-level language modeling, perplexity on the standard test set.

**Why this benchmark.** Bai-Koltun-Kolter NeurIPS 2019 introduced DEQ Transformers on WikiText-103, demonstrating DEQs competitive with explicit transformers. monDEQ has not been seriously deployed at LM scale — this is partly because the expressivity restriction bites harder for language than for vision.

**Architecture.**
- Encoder $\varphi$: standard transformer encoder (word embeddings + positional + 2 transformer blocks).
- Equilibrium state: token-level representations of dimension $n = 512$.
- Monotone core: P1 applied per token, with cross-token attention factored in via a separate (non-monotone) attention sublayer — see implementation notes below.
- Flow $g$: invertible 1×1 conv applied across feature dim, per token.
- Head: standard LM head (linear projection to vocab).

**Implementation note.** Pure monotone attention is not straightforward — attention is naturally non-monotone. Two options:
1. **Hybrid:** apply monotone Linearizer only to the feed-forward sublayer; keep standard attention. The monotonicity guarantee then applies to the FFN-equivalent component only. This is the conservative choice.
2. **Full monotone:** parameterize attention as a monotone operator via Baker et al. 2023 IGNN constructions. More ambitious; defer to v2.

Default: hybrid.

**Baselines.**
- DEQ Transformer (Bai-Koltun-Kolter 2019).
- Standard Transformer (small, matched parameter count).
- monDEQ Transformer (apples-to-apples; we may need to implement this since the original paper didn't).
- RevDEQ (Sept 2025).

**Metrics.**
- Validation perplexity.
- Test perplexity.
- Equilibrium error.
- Training stability (seeds).

**Expected outcome.** Match DEQ Transformer perplexity (~24 on WikiText-103 with this scale). Demonstrate monDEQ underperforms substantially in the hybrid setting (expected perplexity ~30+ with the spectral constraint).

### 5.3 Application 3 — Long-range graph reasoning (OGB-LSC, IGNN benchmarks)

**Task:** Graph property prediction tasks that require long-range information flow. Baker-Wang-Hauck-Wang ICML 2023 established these as the natural benchmark for implicit GNNs.

**Datasets:**
- **ogbn-arxiv** (node classification, 169k nodes).
- **PPI** (multi-label node classification).
- **PascalVOC-SP, COCO-SP** (long-range graph benchmark; Dwivedi et al. 2022).

**Why this benchmark.** Implicit GNNs (IGNNs) are the natural application of monotone operator theory to graphs. Baker et al. 2023 already addressed monDEQ's expressivity in the IGNN setting; we directly compare against their relaxed parameterization with a different (Linearizer-style) lifting.

**Architecture.**
- Encoder $\varphi$: node feature embeddings + 1-2 GNN layers (GCN or GAT).
- Latent dim $n$ = (node count × feature dim). Use sparse representation.
- Monotone core: P2 (Baker et al. parameterization) applied as graph convolution with monotonicity constraint.
- Flow $g$: node-level invertible flow (shared across nodes).
- Head: node-level or graph-level prediction head.

**Baselines.**
- IGNN (Gu et al. 2020): original implicit GNN.
- Monotone IGNN (Baker et al. 2023): direct comparison.
- Standard GNN baselines (GCN, GAT, GIN).
- Long-range MPNN baselines (e.g., GraphTransformer).

**Metrics.**
- Task-specific (accuracy, F1, MRR).
- Long-range scoring per Dwivedi 2022 protocol.
- Equilibrium convergence behavior on graphs with varying diameter.

**Expected outcome.** Match or beat Baker et al. 2023 results, especially on graphs with large diameter where monotone equilibrium models have a structural advantage.

### 5.4 Cross-application: the Pareto headline figure

This is the central figure of the paper, analogous to the tropical blueprint's Kleene-collapse figure.

**Experiment.** On a fixed task (CIFAR-100 recommended), produce a 2D scatter plot:
- $x$-axis: forward iterations to convergence (stability proxy).
- $y$-axis: accuracy (expressivity proxy).

Plot points:
- Standard DEQ (high $y$, high $x$, with error bars indicating training instability).
- monDEQ (low $y$, low $x$).
- ResNet (no equilibrium, fixed $x$, varying $y$).
- Monotone Linearizer DEQ (high $y$, low $x$) — Pareto-dominant.

**Expected figure.** Our point sits in the upper-left (high accuracy, fast convergence), Pareto-dominating monDEQ and tying or beating DEQ in expressivity while retaining monDEQ's stability. Make this the visual identity of the paper.

---

## 6. Implementation details

### 6.1 Repository structure (for Claude Code)

```
monotone-linearizer/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── cifar.yaml
│   ├── wikitext.yaml
│   └── ogb.yaml
├── monotone_linearizer/
│   ├── __init__.py
│   ├── monotone/
│   │   ├── parameterizations.py    # P1 (Winston-Kolter), P2 (Baker), P3 (ICNN)
│   │   ├── solvers.py              # forward-backward, Peaceman-Rachford
│   │   ├── implicit_diff.py        # implicit function theorem for backprop
│   │   └── core.py                 # MonotoneCore module
│   ├── flows/
│   │   ├── coupling.py             # affine coupling
│   │   ├── glow.py                 # 1x1 conv, actnorm, multi-scale
│   │   ├── coupling_seq.py         # sequence-aware coupling for LM
│   │   └── invertible.py           # InvertibleFlow wrapper
│   ├── models/
│   │   ├── monotone_linearizer.py  # MonotoneLinearizer = head ∘ g⁻¹ ∘ M_solve ∘ φ
│   │   ├── ml_classifier.py        # for vision
│   │   ├── ml_transformer.py       # for LM
│   │   └── ml_gnn.py               # for graphs
│   ├── data/
│   │   ├── cifar.py
│   │   ├── wikitext.py
│   │   └── ogb.py
│   ├── train/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── stability.py            # Jacobian regularization, equilibrium tolerance scheduling
│   │   └── optim.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── equilibrium.py          # convergence diagnostics
│   │   └── stability.py            # seed variance, divergence detection
│   └── utils/
│       ├── logging.py
│       └── viz.py                  # Pareto plot helpers
├── scripts/
│   ├── train_cifar.py
│   ├── train_wikitext.py
│   ├── train_ogb.py
│   ├── pareto_plot.py              # produces headline figure
│   └── reproduce_table_*.py
├── tests/
│   ├── test_monotonicity.py        # verify M is m-strongly monotone by construction
│   ├── test_invertibility.py       # g(g⁻¹(x)) = x
│   ├── test_fixed_point.py         # equilibrium error after solving
│   ├── test_implicit_diff.py       # gradient correctness
│   └── test_composition.py         # monotone composition under cocoercivity
└── paper/
    ├── main.tex
    ├── sections/
    └── figs/
```

### 6.2 Key implementation notes

**Monotone parameterization P1 (Winston-Kolter).**

```python
class MonotoneCore(nn.Module):
    """W parameterized so I - W ⪰ mI is satisfied by construction."""
    def __init__(self, n, m=0.1, hidden=None):
        super().__init__()
        self.n, self.m = n, m
        self.A = nn.Parameter(torch.randn(n, n) * 0.01)  # unconstrained
        self.U = nn.Linear(hidden or n, n, bias=True)    # input injection
        
    def W(self):
        I = torch.eye(self.n, device=self.A.device)
        return (1 - self.m) * I - self.A.T @ self.A      # I - W = mI + A^T A ⪰ mI ✓
        
    def forward(self, w, c):
        return F.relu(self.W() @ w + self.U(c))
```

**Test: verify monotonicity numerically.** For random $u, v$ and random $c$:
```python
def test_monotone():
    core = MonotoneCore(n=64, m=0.1)
    for _ in range(100):
        u = torch.randn(64); v = torch.randn(64); c = torch.randn(...)
        G_u = u - core(u, c); G_v = v - core(v, c)
        assert torch.dot(G_u - G_v, u - v) >= core.m * torch.norm(u - v)**2 - 1e-6
```

**Solver: Peaceman-Rachford splitting.**

```python
def peaceman_rachford(core, c, w_init, n_iter=50, tol=1e-4):
    """Solve w = core(w, c) via PR splitting."""
    w = w_init
    for k in range(n_iter):
        # ... standard PR splitting iteration (see Winston-Kolter Alg. 1)
        if equilibrium_error(w) < tol:
            break
    return w
```

Use the reference implementation from the monDEQ repo (Winston-Kolter 2020 released code) as a starting point.

**Implicit differentiation.** Use the standard DEQ implicit-diff trick: in forward, run the solver to convergence (no grad); in backward, solve $(I - \partial_w M)^{-1} v$ once with GMRES.

```python
class MonotoneFixedPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, c, w_init):
        with torch.no_grad():
            w_star = peaceman_rachford(M, c, w_init)
        ctx.save_for_backward(w_star, c)
        ctx.M = M
        return w_star
    
    @staticmethod
    def backward(ctx, grad_output):
        w_star, c = ctx.saved_tensors
        M = ctx.M
        # solve (I - J_w M)^T v = grad_output
        v = solve_implicit(M, w_star, c, grad_output)
        # propagate to inputs
        grad_c = torch.autograd.grad(M(w_star, c), c, v)[0]
        return None, grad_c, None
```

### 6.3 Hyperparameter defaults

| Parameter | Default | Notes |
|---|---|---|
| Latent dim $n$ | 256 (vision) / 512 (LM) / variable (graph) | |
| Monotonicity $m$ | 0.1 | Higher = more stable, less expressive |
| Flow blocks $K$ | 6 | Match original Linearizer |
| Solver iterations max | 50 | With early-stopping on equilibrium error |
| Equilibrium tolerance | 1e-3 (training) → 1e-5 (eval) | Schedule during training |
| Jacobian reg weight | 1e-4 | From Bai-Koltun-Kolter 2021 |
| Optimizer | AdamW, lr 5e-4 (vision) / 1e-4 (LM) | |
| Batch size | 128 (vision) / 32 (LM) / variable (graph) | |
| Training steps | 100k (vision), 200k (LM), 50k (graph) | |

### 6.4 Compute budget

Per-experiment estimates (single A100):
- CIFAR-10/100: ~12-24 hours.
- WikiText-103: ~48-72 hours.
- OGB tasks: ~6-12 hours each.

Total for full table reproduction: ~2 GPU-weeks per dataset family with ablations. Budget ~4 GPU-weeks total.

---

## 7. Ablations

### 7.1 Critical ablations (must include — these test the central claim)

**A1. $g = \text{Id}$ (the *non-trivial g earns its keep* test).** Set the flow to identity. This recovers exact monDEQ. Expected: significantly underperforms full Monotone Linearizer on CIFAR-100 (≥3% accuracy gap) and WikiText (≥5 perplexity gap). **This is the most important ablation in the paper.** If A1 doesn't show a substantial gap, the paper's central claim is false and we need to know.

**A2. Trivial $g$ (linear, fixed).** Set $g$ to a fixed random orthogonal matrix. Tests whether the *learning* of $g$ matters, not just the presence of a coordinate change. Expected: intermediate result between A1 and full Monotone Linearizer.

**A3. Flow depth.** Vary $K \in \{1, 2, 4, 6, 8\}$. Expected: depth matters up to ~4-6, then plateaus.

**A4. Monotonicity parameter $m$.** Sweep $m \in \{0.01, 0.05, 0.1, 0.3, 0.5\}$. Expected: low $m$ gives best accuracy but slowest convergence; high $m$ converges fast but limits expressivity. Show this tradeoff curve as a secondary figure.

**A5. Parameterization comparison.** P1 vs. P2 vs. P3. Expected: P1 strongest for simple tasks, P2 best for graphs, P3 best when convex potentials are natural.

**A6. Flow type.** Glow vs. RealNVP vs. NSF (neural spline flows). Expected: minor differences; Glow is the simplest viable choice.

**A7. Without monotonicity (DEQ baseline).** Replace M with unconstrained operator (= standard DEQ). Show: high expressivity but training divergence rate increases sharply. Quantify: fraction of seeds that diverge during training.

### 7.2 Secondary ablations

**B1. Effect of input encoder $\varphi$ depth.**
**B2. Solver choice:** Peaceman-Rachford vs. forward-backward vs. Anderson acceleration.
**B3. Equilibrium tolerance during training.**
**B4. Jacobian regularization weight.**
**B5. Cocoercivity in composition** (two stacked Monotone Linearizer layers): does naive stacking preserve monotonicity in latent? Expected: only under cocoercivity assumptions; document failure modes.

---

## 8. Figures and tables

**Figures:**

1. **Fig. 1 — Architectural diagram.** Top: input → encoder → latent monotone solver → $g^{-1}$ → head. Bottom: visualization of "monotonicity in induced metric" with simple 2D example.
2. **Fig. 2 — Pareto headline (HEADLINE).** Accuracy vs. iterations-to-convergence scatter, with monDEQ, DEQ, Monotone Linearizer, and a few baselines. Our point Pareto-dominant.
3. **Fig. 3 — Training stability.** Loss curves across seeds for DEQ vs. monDEQ vs. ours. Show: ours has monDEQ's tight band, DEQ's height.
4. **Fig. 4 — Equilibrium convergence behavior.** Plot $\|w^k - w^{k-1}\|$ vs. iteration $k$ for the three methods on a held-out batch.
5. **Fig. 5 — Effect of $m$.** Accuracy and convergence-rate vs. $m$, showing tradeoff frontier.
6. **Fig. 6 — Graph long-range performance.** Accuracy vs. graph diameter, showing our method's advantage on large-diameter graphs.

**Tables:**

1. **Tab. 1 — CIFAR-10/100.** Accuracy, parameters, convergence iterations, training divergence rate. Compare against full DEQ literature.
2. **Tab. 2 — WikiText-103.** Test perplexity, parameters, convergence behavior.
3. **Tab. 3 — OGB / long-range graph tasks.** Per-task metrics, compare against IGNN, MonotoneIGNN (Baker 2023), GraphTransformer.
4. **Tab. 4 — Ablation summary.** A1-A7 with primary metric on CIFAR-100.
5. **Tab. 5 — Hyperparameter sensitivity.** Heatmap of accuracy across $(m, K)$ grid.

---

## 9. Related work — precise differentiation

**Original Linearizer (Shocher-Berman-Hallak 2025).**
> "We extend the Linearizer framework from the linear semiring core to a monotone-operator core. The induced inner product machinery transfers cleanly, giving monotonicity-in-induced-metric as the central structural property. Composition closure requires additional cocoercivity assumptions (versus automatic in the linear case). Applications shift from generative modeling (one-step flow matching, IGN) to implicit deep learning (DEQ models with guaranteed equilibria)."

**Deep Equilibrium Models (Bai-Koltun-Kolter NeurIPS 2019, ICML 2021).**
> "DEQs introduced the implicit-depth paradigm but acknowledge instability and lack of equilibrium uniqueness guarantees. We retain the DEQ machinery (implicit differentiation, equilibrium solvers) but constrain the operator structure via monotonicity in learned coordinates."

**Monotone Operator Equilibrium Networks (Winston-Kolter NeurIPS 2020).**
> "monDEQ is the foundation we build on. Our contribution is to lift monDEQ's expressivity restriction by transporting the monotonicity property into a learned coordinate system. The monDEQ paper itself acknowledges the expressivity cost of the spectral constraint, motivating our approach. Ablation A1 ($g = \text{Id}$) reduces to monDEQ exactly, providing the cleanest possible apples-to-apples comparison."

**Implicit GNNs via Monotone Operators (Baker-Wang-Hauck-Wang ICML 2023).**
> "Baker et al. also address monDEQ's expressivity restriction, but via a different mechanism: a relaxed parameterization that preserves well-posedness through a different monotonicity structure. We pursue a complementary approach (coordinate transport via invertible flow). The two approaches can be composed — our flow on top of their parameterization (P2) — and we report on this combination in Application 3."

**Reversible DEQs (RevDEQ, Sept 2025).**
> "RevDEQ uses reversibility for exact gradient computation in DEQs, motivated by training stability. Our use of invertible flows is structurally similar (both employ reversible architectures) but motivationally distinct: we use invertibility to lift expressivity restrictions on the monotone core, not for gradient exactness. The two contributions are composable."

**Monotone operators for inverse problems (Pesquet et al. 2021; Belkouchi et al. 2024; Repetti et al. 2025).**
> "These works learn monotone operators (sometimes via soft penalization, sometimes via graph convergence) for use as plug-and-play priors in inverse problems. We provide a complementary by-construction architecture; our flow-based lifting could be combined with their training procedures. We do not include inverse problems in the main experimental matrix (defer to follow-up) but note the natural extension."

**Monotonic neural networks (Sill 1996; Daniels-Velikova 2010; Runje-Shankaranarayana 2023; Nolte-Kitouni-Williams 2023; Sartor et al. 2025).**
> "These works build neural networks that are monotone in the *coordinate-wise input-output sense* (output increases with each input), used for enforcing domain knowledge in tabular regression. This is a fundamentally different sense of 'monotone' from operator-theoretic monotonicity. Our framework targets the operator-theoretic sense and is not applicable to the input-monotonic use case (a learned flow $g$ would in fact destroy coordinate-wise input monotonicity)."

**Normalizing flows (Dinh et al. 2015, 2017; Kingma-Dhariwal Glow 2018).**
> "We use normalizing-flow architectures as building blocks for $g$ to transport algebraic structure, not probability measures. The Jacobian determinant is not used in our objective."

---

## 10. Limitations (be honest in the paper)

1. **Composition closure is conditional.** Composition of monotone operators is monotone only under cocoercivity (or stronger) assumptions. Stacking Monotone Linearizer layers requires care; we discuss in ablation B5 and note as a limitation versus the linear Linearizer (where composition is automatic). 

2. **Monotonicity holds in induced metric, not standard metric.** For applications that require standard-metric monotonicity (e.g., proximal algorithms with Euclidean steps), our framework is not directly applicable. We use latent-space iteration to sidestep this; we don't claim our $f$ is monotone in standard Euclidean inner product.

3. **No automatic SVD / spectral theory analog.** The linear Linearizer's SVD lemma doesn't have a clean monotone-operator counterpart. We provide a resolvent analog instead (see §3.4).

4. **Flow computational cost.** The flow $g$ adds inference cost (one forward + one inverse per sample). For very large-scale tasks, this could be a bottleneck.

5. **Restricted to maximally monotone operators.** Operators that are monotone but not maximally monotone (which can happen with non-surjective domain) fall outside our framework. In practice this is rare for the parameterizations we use.

6. **Universal approximation status is open.** Whether $g^{-1} \circ M \circ g$ can approximate arbitrary functions (with $M$ in P1 and $g$ an arbitrary flow) is plausible but unproven. We do not claim a UA theorem.

7. **No theoretical proof of the "non-trivial g earns its keep" claim.** §3.3 states the claim informally. The formal version requires constructing a concrete separating example. We rely on the empirical A1 ablation as the operational proxy.

---

## 11. Risks and mitigations

**R1. Ablation A1 doesn't show a meaningful gap.** Highest-impact risk. If $g = \text{Id}$ already gets us most of the accuracy, the paper has no contribution. *Mitigation:* run A1 on CIFAR-100 in the first 2 weeks of execution. If the gap is < 1%, pivot immediately — either to a different application where the gap is larger, or to a different framework.

**R2. monDEQ baseline is too strong on our chosen benchmarks.** Baker et al. 2023 already addressed expressivity in graphs. *Mitigation:* compare against their parameterization (P2) head-to-head, not just original monDEQ. Frame the contribution as complementary lifting rather than replacement.

**R3. RevDEQ collision.** Recent paper uses invertible architecture in DEQs. *Mitigation:* differentiate clearly — RevDEQ is about gradient exactness, we are about expressivity lifting via monotonicity guarantees. Different motivations, different proofs, composable contributions.

**R4. Standard DEQ has been engineered hard and is hard to beat.** *Mitigation:* match standard DEQ accuracy (don't claim to beat) while substantially improving stability metrics (divergence rate, seed variance). This is still a clear Pareto improvement and a publishable result.

**R5. Training is slow due to flow + DEQ + implicit diff combination.** *Mitigation:* use small flow (K = 6), implement efficient batched solver, profile early. If training is impractical, reduce latent dim or simplify flow.

**R6. The technical machinery (induced inner product + monotone splitting) is intimidating for the reader.** *Mitigation:* lean on Lemma 4 (latent-space computation is just standard monDEQ in Euclidean) to keep the operational story simple. Move Hilbert-space machinery to appendix.

---

## 12. Writing plan

**Section structure (target 10 pages main + appendix):**

| Section | Pages | Notes |
|---|---|---|
| 1. Introduction | 1.0 | Lead with Pareto-tradeoff pitch |
| 2. Background | 0.75 | DEQ + monDEQ + Linearizer recap |
| 3. Monotone Linearizer framework | 1.75 | Defs 1-2, Lemmas 1-5, §3.3 informal theorem |
| 4. Architecture & training | 0.75 | P1 default, flow, solver |
| 5. App 1: CIFAR-10/100 | 1.5 | Pareto figure here |
| 6. App 2: WikiText-103 | 1.0 | LM perplexity table |
| 7. App 3: Long-range graphs | 1.0 | OGB + IGNN comparison |
| 8. Analysis & ablations | 1.0 | A1-A7 summary |
| 9. Related work | 0.5 | Per §9 above |
| 10. Limitations & conclusion | 0.75 | Honest |
| Appendix | — | Full proofs, induced-metric machinery, additional experiments |

**Writing order:**

1. Framework section (§3) first — lock down theory.
2. App 1 CIFAR runs — produce the Pareto headline figure.
3. Ablation A1 — confirm the central claim empirically. **STOP if this fails.**
4. App 2 and App 3 in parallel.
5. Related work and Limitations after results are in.
6. Introduction last.

---

## 13. Collaborators and division of labor

**Lead:** Assaf Hallak. Theory, ablations, headline figure.

**Suggested ask list (resolve via NVIDIA IP channels per past discussion):**

- **Assaf Shocher** (original Linearizer first author). Strongest natural co-author given the framework continuity. The Linearizer machinery is already well-understood between you; the monotone extension is a clean continuation.
- **Nimrod Berman** (KDM + Linearizer). Could own implementation of the flow + solver integration.
- **Bai, Koltun, Kolter** — *external sanity check only*, not co-authors unless they actively engage. Their work is the foundation we build on; courtesy outreach after a working prototype is appropriate.
- **NVIDIA team (Eli Meirom, Yuval Atzmon, Chen Tessler):** Application 3 (graphs) is a natural fit; Eli's prior graph work could shape the OGB experiment design.

**Open question.** Same as last time: how does this fit your RL-adjacent NVIDIA mandate? Monotone Linearizer is closer to representation learning and implicit models than to RL. If RL/LLM mandate is strict, this is a parallel project; if there's flexibility, it could be a primary push.

---

## 14. Reproducibility checklist

- [ ] All code under Apache 2.0 on GitHub.
- [ ] Pretrained checkpoints for all reported numbers.
- [ ] Hydra configs for every experiment.
- [ ] `scripts/reproduce_table_N.py` for each table.
- [ ] Unit tests verifying monotonicity (P1, P2, P3), invertibility, equilibrium error, implicit-diff correctness.
- [ ] WandB project link.
- [ ] CIFAR, WikiText, OGB datasets from official repos; no modifications.
- [ ] Ablation A1 has its own clearly-labeled script and result so reviewers can verify the central claim.

---

## 15. Decision points for Assaf (resolve before kickoff)

1. **Scope commitment.** Primary push for ICLR 2027 or parallel thread? Estimate: 4 GPU-weeks + 6-8 weeks focused work.

2. **Co-authors.** Shocher in or out? Decision affects framing (sequel paper vs. independent extension).

3. **Application prioritization.** If forced to drop one: Application 2 (WikiText) is most replaceable — the monotone-attention hybrid story is the most speculative. Recommendation: keep Apps 1 and 3 as primary, App 2 as confirmation if time permits.

4. **Hybrid vs. full-monotone attention (App 2).** Default: hybrid (monotone FFN, standard attention). Full-monotone attention is a v2 contribution. Confirm OK.

5. **Inverse problems application — include or defer?** Pesquet-line work is a natural fit. Decision: include only if execution timeline allows. Default: defer.

6. **Universal approximation theorem.** Pursue or skip? Recommendation: skip for this paper, note as future work. Could be a separate theory paper (Nadav collaboration).

7. **Composition closure subtlety.** Discuss in main paper or appendix? Recommendation: main paper, brief acknowledgment in §3.2, full discussion in appendix. Reviewers will ask.

---

## 16. Bibliography seeds (for `paper/refs.bib`)

**Core (Linearizer + this work):**
- Shocher, Berman, Hallak. "Who Said Neural Networks Aren't Linear?" ICLR 2026 (under review).
- This paper.

**DEQ and implicit models:**
- Bai, Kolter, Koltun. "Deep Equilibrium Models." NeurIPS 2019.
- Bai, Koltun, Kolter. "Multiscale Deep Equilibrium Models." NeurIPS 2020.
- Bai, Koltun, Kolter. "Stabilizing Equilibrium Models by Jacobian Regularization." ICML 2021.
- Gurumurthy, Bai, Manchester, Kolter. "Joint Inference and Input Optimization in Equilibrium Networks." NeurIPS 2021.
- El Ghaoui, Gu, Travacca, Askari, Tsai. "Implicit Deep Learning." SIAM J. Math. Data Sci. 2021.
- Reversible DEQ (RevDEQ). 2025. arXiv:2509.12917.
- Tsuchida, Ong. "Deep Equilibrium Models as Estimators for Continuous Latent Variables." 2022.

**Monotone operator theory in deep learning:**
- Winston, Kolter. "Monotone Operator Equilibrium Networks." NeurIPS 2020.
- Pabbaraju, Winston, Kolter. "Estimating Lipschitz Constants of Monotone Deep Equilibrium Models." ICLR 2021.
- Revay, Wang, Manchester. "Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness." 2023.
- Baker, Wang, Hauck, Wang. "Implicit Graph Neural Networks: A Monotone Operator Viewpoint." ICML 2023.
- Pesquet, Repetti, Terris, Wiaux. "Learning maximally monotone operators for image recovery." SIAM J. Imaging Sci. 2021.
- Belkouchi, Pesquet, Repetti, Terris. "Learning truly monotone operators." 2024.
- Approximation of maximally monotone operators (Repetti et al.). 2025. arXiv:2605.12301.

**Monotone operator theory (foundations):**
- Bauschke, Combettes. *Convex Analysis and Monotone Operator Theory in Hilbert Spaces*. Springer, 2017.
- Ryu, Boyd. "Primer on Monotone Operator Methods." 2016.

**Normalizing flows:**
- Dinh, Krueger, Bengio. "NICE." 2015.
- Dinh, Sohl-Dickstein, Bengio. "Density estimation using Real NVP." ICLR 2017.
- Kingma, Dhariwal. "Glow." NeurIPS 2018.
- Rezende, Mohamed. "Variational Inference with Normalizing Flows." ICML 2015.

**Benchmarks:**
- CIFAR-10/100 (Krizhevsky 2009).
- WikiText-103 (Merity et al. 2016).
- OGB (Hu et al. 2020).
- Long-range graph benchmark (Dwivedi et al. 2022).

**Input-monotonic NNs (for related work differentiation):**
- Sill. "Monotonic Networks." NeurIPS 1996.
- Daniels, Velikova. "Monotone and Partially Monotone Neural Networks." IEEE TNN 2010.
- Runje, Shankaranarayana. "Constrained Monotonic Neural Networks." ICML 2023.
- Nolte, Kitouni, Williams. "Expressive Monotonic Neural Networks." ICLR 2024.
- Sartor et al. "Advancing Constrained Monotonic Neural Networks." ICML 2025.

**ICNN (for parameterization P3):**
- Amos, Xu, Kolter. "Input Convex Neural Networks." ICML 2017.

---

## 17. Implementation milestones (6-8 week plan)

**Week 1: Foundations.**
- Repo scaffold, monotone parameterizations P1-P3 with unit tests verifying $m$-strong monotonicity numerically.
- Solver (Peaceman-Rachford) with convergence tests.
- Implicit differentiation with gradient correctness tests.
- Invertible flow with invertibility tests.

**Week 2: Integration + CIFAR pilot.**
- Wire flow + monotone solver + head together.
- Train on CIFAR-10 with small architecture.
- Verify equilibrium error stays bounded and accuracy is non-trivial.

**Week 3: Ablation A1 (CRITICAL).**
- Run A1 ($g = \text{Id}$) on CIFAR-100.
- **Decision gate:** if A1 gap < 1%, halt and pivot. If gap ≥ 3%, proceed.
- Headline Pareto figure first draft.

**Week 4: CIFAR-100 full + ablations A2-A6.**
- Scale to CIFAR-100, full training.
- Run ablations.
- Tab. 1 and Fig. 2 finalized.

**Week 5: WikiText-103 (App 2).**
- Hybrid monotone-FFN + standard attention DEQ Transformer.
- Train on WikiText-103.
- Tab. 2 finalized.

**Week 6: OGB / IGNN (App 3).**
- Implement P2 (Baker et al. parameterization).
- Train on ogbn-arxiv, PascalVOC-SP.
- Tab. 3 finalized.

**Week 7: Polish + paper writing.**
- Remaining ablations, qualitative analysis.
- Paper draft sections complete.
- Internal review.

**Week 8: Buffer + submission.**
- Final figures, last experiments, camera-ready prep.

---

## 18. Quick gut check — should we actually do this?

**Reasons yes:**
- The "non-trivial g earns its keep" test passes cleanly: with $g = \text{Id}$ we recover monDEQ (acknowledged restrictive); with learned $g$ we recover expressivity. Falsifiable, central, testable in week 3.
- The wish-fulfillment story is genuine and widely shared: "if only this DEQ were stable with unique equilibria."
- Vector-valued throughout — no scalar degeneracy.
- Composition closure (with caveats), induced inner product, resolvent — the Linearizer framework transfers in a meaningful way.
- Multiple established benchmarks (CIFAR, WikiText, OGB) with strong baseline implementations available.
- The Pareto-improvement story (match DEQ accuracy, match monDEQ stability) is clean and visible in one figure.

**Reasons to slow down:**
- DEQ space is saturated. Bai/Koltun/Kolter line + Baker et al. + RevDEQ + Pesquet line means many strong groups in the room. The bar is high.
- Implementation overhead is real — flow + DEQ + implicit diff is three pieces to integrate.
- Composition closure is conditional, not automatic. This is a real theoretical weakness versus the linear Linearizer.
- Application 2 (LM) requires the hybrid monotone-attention design which is the most speculative piece.

**Recommendation.** This is a strong empirical paper if executed cleanly. The critical filter is week-3 ablation A1: if the non-trivial $g$ doesn't earn its keep on CIFAR-100, kill the project and pivot. If A1 succeeds, the rest is engineering.

**Compared to tropical:** Monotone Linearizer has a more compelling wish-fulfillment story and a more directly competitive empirical setup (established DEQ benchmarks with clear baselines). It is a harder paper to write well because the area is saturated, but the upside is correspondingly larger.

---

*End of blueprint. Hand off to Claude Code with the prompt: "Implement the Monotone Linearizer following this blueprint. Execute Weeks 1-3 in order; halt for review after ablation A1 (the critical gate)."*
