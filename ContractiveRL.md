# Contractive Linearizer for RL — PoC Specification

## Goal

Implement and evaluate a **contractive Linearizer-based iterative solver** for reinforcement learning and planning.

We aim to test whether enforcing **contraction by architecture** leads to:
- stable convergence
- better planning behavior
- robustness vs standard neural iterative methods (VIN / MLP baselines)

---

## Core Idea

We model an **iterative operator**:

x_{k+1} = T_\theta(x_k, c)

Where:
- x = state representation (e.g., value map, Q-values, latent plan)
- c = context (environment, observation)

We define:

T_\theta(x, c) = g^{-1}( A(c) * g(x) )

Where:
- g: invertible neural network (encoder)
- g^{-1}: decoder
- A(c): context-dependent linear operator
- A(c) is **contractive**: spectral radius < 1

---

## Key Properties

1. **Guaranteed convergence**:
   - Iteration converges to fixed point

2. **Controllable convergence speed**:
   - Eigenvalues of A(c)

3. **Fast multi-step simulation**:
   - x_K = g^{-1}( A(c)^K * g(x_0) )

---

## Tasks to Implement

---

## 1. Core Modules

### 1.1 Invertible Network g

Implement a simple invertible architecture:

Options:
- NICE / RealNVP style affine coupling
- Simpler: reversible residual block (if easier)

API:
- encode(x) → z
- decode(z) → x

---

### 1.2 Contractive Operator A(c)

Implement A(c) with guaranteed contraction.

Options (start simple):

#### Option A: Diagonal
A(c) = diag( tanh(s(c)) )

#### Option B: Low-rank
A(c) = U(c) V(c)^T
Then scale:
A(c) ← A(c) / (1 + ||A(c)||)

#### Option C: Spectral parameterization
A(c) = Q Λ Q^{-1}
Where:
- Λ = diag(σ_i), σ_i ∈ (-1, 1)

---

### 1.3 Iteration Engine

Function:

iterate(x0, c, K):
    z0 = g(x0)
    zK = (A(c) ^ K) @ z0
    return g^{-1}(zK)

Also implement:
- naive iteration (loop K times)
- fast exponentiation (matrix power)

---

## 2. Environments

Start with simple deterministic environments:

### 2.1 Gridworld (MANDATORY)
- size: 10x10 or 20x20
- goal state
- obstacles
- reward = -1 per step, 0 at goal

Represent:
- x = value map (grid)
- c = map encoding

---

### 2.2 Optional
- MiniGrid (if time permits)

---

## 3. Training Setup

---

### 3.1 Value Iteration Target

Ground truth:
- compute optimal value function V*

Train T to satisfy:

T(V, c) ≈ Bellman update

Loss:

L = || T(V, c) - Bellman(V, c) ||^2

---

### 3.2 Fixed Point Training (IMPORTANT)

Also train toward fixed point:

L_fp = || T(V*, c) - V* ||^2

---

### 3.3 Multi-step Consistency

L_multi = || T^K(V0, c) - V* ||^2

Test multiple K values.

---

## 4. Baselines

Implement:

### 4.1 MLP
- direct value prediction

### 4.2 VIN (Value Iteration Network)
- standard convolutional planner

### 4.3 Unconstrained Linearizer
- same model without contraction constraint

---

## 5. Experiments

---

## 5.1 Convergence Test (CRITICAL)

For each model:

- initialize random V0
- iterate K steps

Measure:
- ||V_k - V*||
- stability (does it diverge?)

Plot:
- error vs iteration

Expected:
- contractive model always converges
- others may oscillate/diverge

---

## 5.2 Planning Quality

Evaluate:
- final policy derived from V
- success rate to reach goal

---

## 5.3 Generalization

Train on:
- small maps

Test on:
- larger maps

Measure:
- performance drop

---

## 5.4 Fast Iteration Trick

Compare:

1. Iterative:
   x_K via loop

2. Collapsed:
   x_K = g^{-1}( A(c)^K g(x0) )

Measure:
- speed
- numerical difference

---

## 5.5 Ablation

Test:
- different spectral bounds
- contraction strength

---

## 6. Metrics

- MSE to V*
- convergence rate
- policy success rate
- runtime (iterative vs collapsed)
- stability (variance across runs)

---

## 7. Expected Outcomes

We are testing if:

1. Contractive model:
   - always converges
   - is stable across seeds

2. Baselines:
   - may diverge or oscillate

3. Collapse trick:
   - matches iterative behavior
   - gives speedup

---

## 8. Minimal Success Criteria

We consider the idea promising if:

- contractive model consistently converges
- baselines sometimes fail
- performance is comparable or better
- collapse trick works numerically

---

## 9. Stretch Goals

- apply to Q-learning
- continuous control toy problem
- learn A(c) interpretable spectrum

---

## Notes

- Start with low-dimensional latent (e.g., 64)
- Keep A(c) small for fast matrix power
- Prefer stability over performance at first

---

## Deliverables

- training scripts
- evaluation plots
- comparison tables
- visualization of convergence
