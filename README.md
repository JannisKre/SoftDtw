# Soft-DTW Ball Solver — Gradient Descent Approach

This repository contains the **soft-DTW gradient descent solver** for the DTW Ball Separation Problem, developed as part of a master's thesis on the VC dimension of DTW-based hypothesis classes.

## Background

Given univariate time series $Q_1, \ldots, Q_s \in \mathbb{R}^m$ and a subset $I \subseteq \{1, \ldots, s\}$, the **DTW Ball Separation Problem** asks:

> Does there exist a center curve $P \in \mathbb{R}^k$ and a radius $\Delta \geq 0$ such that $\mathrm{DTW}(P, Q_i) \leq \Delta$ for all $i \in I$, and $\mathrm{DTW}(P, Q_j) > \Delta$ for all $j \notin I$?

If every subset $I \subseteq \{1, \ldots, s\}$ is realizable, then the $s$ query curves are **shattered** by DTW balls of center-complexity $k$, giving a lower bound on the VC dimension of the range space $\mathcal{R}^1_{k,m}$.

## Approach: Soft-DTW + Gradient Descent

The standard DTW distance is not differentiable (it takes a hard minimum over warping paths), making gradient-based optimization impossible. This solver replaces DTW with the **soft dynamic time warping distance** of Cuturi & Blondel (2017):

$$\mathrm{dtw}^\gamma(P, Q) = \min^\gamma_{A \in \mathcal{A}_{k,m}} \langle A,\, \Delta(P,Q) \rangle$$

where $\min^\gamma(a_1, \ldots, a_n) = -\gamma \log \sum_i e^{-a_i/\gamma}$ is the soft minimum. As $\gamma \to 0$, this converges to the standard DTW distance; for $\gamma > 0$ it is smooth and differentiable everywhere.

### Optimization procedure

For a fixed labeling $I$, a center $P \in \mathbb{R}^k$ and threshold $\Delta$ are optimized simultaneously by minimising the **hinge separation loss**:

$$L_I(P, \Delta) = \sum_{i \in I} \max\{0,\, \mathrm{dtw}^\gamma(Q_i, P) - \Delta + \eta\} + \sum_{j \notin I} \max\{0,\, \Delta - \mathrm{dtw}^\gamma(Q_j, P) + \eta\}$$

where $\eta > 0$ is a safety margin. A zero loss certifies strict separation with margin $\eta$. Optimization uses **Adam** with multiple random restarts; initialization is the mean of the inside curves resampled to length $k$.

Each gradient step costs $O(RTskm)$ for $R$ restarts, $T$ epochs, and $s$ curves. While still exponential in $s$ (one solve per subset), this approach scales to much larger $k$ and $m$ than the exact QP solver.

**Strengths:** Differentiable, GPU-compatible, handles instances where the QP is too slow.  
**Limitations:** Approximate — can miss separations with very small margins (order $\varepsilon \sim 10^{-4}$).

## Repository Structure

```
SoftDtw/
├── soft_dtw_solver.py            # Core: SoftDTW, loss, optimizer, shattering test
├── run_growth_function.py        # Growth function estimation runner (CLI)
├── run_sequential_k_equals_m.py  # Sequential capacity runner for k=m regime
├── benchmark_separator_runtime.py# Runtime benchmark script
├── requirements.txt              # GPU dependencies (PyTorch)
├── requirements-cpu.txt          # CPU-only dependencies
│
├── soft_dtw_solver.ipynb         # Full solver walkthrough with visualisations
├── soft_dtw_balls.ipynb          # Soft-DTW ball geometry and parameter study
├── synthetic_separator_demo.ipynb# Demo: synthetic separation task (k=m=15)
├── sequential_k_equals_m.ipynb   # Sequential capacity results for k=m
├── growth_function_estimation.ipynb  # Growth function lower bounds
├── test_shattering_k2_m7.ipynb   # Case study: k=2, m=7
├── runtime_benchmark_grid.ipynb  # Runtime analysis across k, m grid
│
├── witnesses_k1_m1.csv           # Curated shattered sets (k=1,m=1)
├── witnesses_k2_m2.csv           #   "              "     (k=2,m=2)
├── witnesses_k3_m3.csv           #   "              "     (k=3,m=3)
├── witnesses_k4_m4.csv           #   "              "     (k=4,m=4)
├── growth_function_results.csv   # Aggregated growth function data (k=m=3..4)
└── k2_m7.csv                     # Shattering witnesses for k=2, m=7
```

## Requirements

### GPU (recommended for large $k$, $m$)

```bash
pip install -r requirements.txt
```

### CPU only

```bash
pip install -r requirements-cpu.txt
```

Core dependencies: `torch >= 2.0`, `numpy >= 1.24`, `pandas >= 2.0`, `matplotlib >= 3.7`.

## Quick Start

### Run the sequential capacity estimator (CLI)

```bash
# Estimate shattering capacity for k=m=3 and k=m=4
python run_sequential_k_equals_m.py --k 3 4 --runs 3
```

### Estimate the growth function

```bash
python run_growth_function.py --k 3 --m 3 --s_max 8 --num_samples 10
```

This samples 10 point sets of each size $s = 1, \ldots, 8$ and counts how many of the $2^s$ subsets are separable, giving a lower bound on the growth function $\Pi_{\mathcal{R}^1_{k,m}}(s)$.

### Programmatic API

```python
from soft_dtw_solver import check_shattering, sequential_capacity_estimation
import numpy as np

# Check whether 4 random points in R^3 are shattered with k=3
points = np.random.randn(4, 3)
result = check_shattering(points, k=3, gamma=0.1, epochs=500, retries=5)
print("Shattered:", result['shattered'])
print("Witnesses:", result['witnesses'])
```

## Computational Results

### k = m regime (equal center and query length)

The gradient descent approach was used to estimate shattering capacity for $k = m \in \{1, \ldots, 9\}$. Results suggest approximately linear growth (roughly $2k$), though random sampling may miss the specific point configurations needed for larger sets.

| $k = m$ | Capacity (soft-DTW) | Capacity (exact QP) |
|---------|--------------------|--------------------|
| 1 | 1 | 1 |
| 2 | 4 | 4 |
| 3 | 6 | 6 |
| 4 | 8 | 8 |
| 5 | ~9 | — |
| 6 | ~9 | — |

### Growth function ($k = m = 3$)

For $k = m = 3$, the growth function $\Pi(s)$ grows as $2^s$ up to $s = 6$, where it falls below $2^s$, confirming that the VC dimension lower bound is at least 6.

### Comparison with the QP solver

The exact QP solver (see [`dtw_intersection`](https://github.com/JannisKre/dtw_intersection)) confirmed all $2^9 = 512$ subsets for a 9-point example ($k=7$, $m=2$), while this solver found witnesses for only 14 — illustrating the trade-off between exactness and scalability. Conversely, for a $k = m = 15$ separation task, the gradient descent approach finds a valid separator while the QP times out.

## Soft-DTW Balls

The `soft_dtw_balls.ipynb` notebook visualises how the **shape of DTW balls changes** with the smoothing parameter $\gamma$:
- At $\gamma \to 0$: the soft-DTW ball converges to the standard DTW ball (union of ellipsoids)
- As $\gamma$ increases: the ball becomes progressively more convex and approaches a Euclidean ball

## Related

- [dtw_intersection](https://github.com/JannisKre/dtw_intersection) — Exact Gurobi-based QP solver; handles small instances with guaranteed correctness.
- [masters-thesis-dtw-vc-dimension](https://github.com/JannisKre/masters-thesis-dtw-vc-dimension) — Master thesis repository: both solvers + visualisations.

## References

- Cuturi, M. & Blondel, M. (2017). *Soft-DTW: a Differentiable Loss Function for Time-Series*. ICML. ([arXiv:1703.01541](https://arxiv.org/abs/1703.01541))
- Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
