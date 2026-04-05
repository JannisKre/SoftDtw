# Soft-DTW Ball Solver — Gradient Descent Approach

This repository contains the **soft-DTW gradient descent solver** for the DTW Ball Separation Problem, developed as part of a master's thesis on the VC dimension of DTW-based hypothesis classes.

## Background

Given univariate time series $Q_1, \ldots, Q_s \in \mathbb{R}^m$ and a subset $I \subseteq \{1, \ldots, s\}$, the **DTW Ball Separation Problem** asks:

> Does there exist a center curve $P \in \mathbb{R}^k$ and a radius $\Delta \geq 0$ such that $\mathrm{DTW}(P, Q_i) \leq \Delta$ for all $i \in I$, and $\mathrm{DTW}(P, Q_j) > \Delta$ for all $j \notin I$?

If every subset $I \subseteq \{1, \ldots, s\}$ is realizable, then the $s$ query curves are **shattered** by DTW balls of center-complexity $k$, giving a lower bound on the VC dimension of the range space $\mathcal{R}^1_{k,m}$.

## ⚠️ Notation Warning — Code vs. Thesis

> **The roles of `k` and `m` in the code are swapped relative to the thesis.**
>
> | Symbol | Thesis meaning | Code meaning |
> |--------|---------------|--------------|
> | `k` | length of center curve $P \in \mathbb{R}^k$ | length of **query** curves $Q_i \in \mathbb{R}^k$ |
> | `m` | length of query curves $Q_i \in \mathbb{R}^m$ | length of **center** curve $P \in \mathbb{R}^m$ |
>
> Consequently, a result file named **`witnesses_k3_m3.csv`** corresponds to
> **thesis parameters $k=3$, $m=3$** (which happen to be the same here), but a
> file `k2_m7.csv` corresponds to **thesis $k=7$, $m=2$**.
> All function arguments, CSV columns, and file names follow **code convention**.

The mathematical notation in this README (Background and formulae) uses **thesis convention**
($k$ = center length, $m$ = query length).

---

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
├── test_shattering_k2_m7.ipynb   # Case study: code-k=2, code-m=7 (thesis k=7, m=2)
├── runtime_benchmark_grid.ipynb  # Runtime analysis across k, m grid
│
├── witnesses_k1_m1.csv           # Curated shattered sets (code k=m=1)   ← thesis data
├── witnesses_k2_m2.csv           #    "             "     (code k=m=2)   ← thesis data
├── witnesses_k3_m3.csv           #    "             "     (code k=m=3)   ← thesis data
├── witnesses_k4_m4.csv           #    "             "     (code k=m=4)   ← thesis data
├── growth_function_results.csv   # Growth function data (code k=m=3)     ← thesis data
├── k2_m7.csv                     # Shattering witnesses (code k=2, m=7 = thesis k=7, m=2)
├── results2/                     # Sequential k=m capacity results k=m=1..9  ← thesis data
│   └── results2/                 #   (nested path; data files are in the inner folder)
└── archived_results/             # Superseded / unused result directories
```

## Result Data Provenance

> Column headings use **code** convention; convert to thesis notation by swapping k↔m.

| File / Directory | Thesis figure | Code `k` | Code `m` | Generated by |
|---|---|---|---|---|
| `witnesses_k*_m*.csv` (root) | "Table of shattered witnesses" | = `m` (diagonal) | = `k` (diagonal) | `sequential_k_equals_m.ipynb` |
| `growth_function_results.csv` | "Growth function Π(s) for k=m=3" | 3 | 3 | `growth_function_estimation.ipynb` |
| `k2_m7.csv` | "9-point witness for thesis k=7, m=2" | 2 | 7 | `test_shattering_k2_m7.ipynb` |
| `results2/results2/` | "Shattering capacity vs. k=m" | 1–9 | 1–9 | `run_sequential_k_equals_m.py` (server run) |
| `archived_results/` | Not used in thesis | — | — | See `archived_results/README.md` |

### CSV Schema

#### `witnesses_k{k}_m{m}.csv` and `results2/results2/witnesses_k*_m*.csv`

Long-format: one row per (run, subset) pair, plus a header row (empty `subset`) per run:

| Column | Type | Description |
|---|---|---|
| `k` | int | **Code** `k` = query-curve length = **thesis** $m$ |
| `m` | int | **Code** `m` = center-curve length = **thesis** $k$ |
| `run_id` | int | Independent repetition index |
| `d_max` | int | Max shattered-set size found in this run |
| `points_json` | JSON array | Shattered point set $Q_1,\ldots,Q_{d_{\max}}$ (on `__POINTS__` header row) |
| `subset` | string | Comma-separated 1-based indices of the "inside" subset; empty = header row |
| `witness_json` | JSON array | Witness center curve $P$ for this subset (length = code `m` = thesis $k$) |
| `delta` | float | Separating radius $\Delta$ |

#### `growth_function_results.csv`

| Column | Description |
|---|---|
| `s` | Sample size (number of query points) |
| `k`, `m` | Code convention — same swap as above |
| `2^s` | Maximum achievable count |
| `pi_hat` | Estimated growth function $\Pi(s)$ (lower bound = max count across samples) |
| `mean_count`, `std_count`, `min_count` | Statistics over sampled point sets |
| `frac_max` | `pi_hat / 2^s` — fraction of subsets realised |
| `fully_shattered_any` | Whether any sample achieved full $2^s$ shattering |
| `all_counts` | JSON list: per-sample counts |
| `gamma`, `epochs`, `retries` | Soft-DTW optimization hyperparameters |
| `sampling_mode`, `sampling_radius_noise`, `num_samples` | Point-sampling configuration |
| `timestamp` | ISO timestamp of the run |

---

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
# Estimate shattering capacity for code-k=m=3 and k=m=4 (thesis k=m same values here)
python run_sequential_k_equals_m.py --k 3 4 --runs 3
```

> **Reminder:** `--k` is the **query**-curve length (thesis $m$) and `--m` is the **center**-curve length (thesis $k$). For the diagonal case $k=m$ the swap is transparent.

### Estimate the growth function

```bash
# code-k=3, code-m=3  →  thesis k=3, m=3
python run_growth_function.py --k 3 --m 3 --s_max 8 --num_samples 10
```

This samples 10 point sets of each size $s = 1, \ldots, 8$ and counts how many of the $2^s$ subsets are separable, giving a lower bound on the growth function $\Pi_{\mathcal{R}^1_{k,m}}(s)$. Results are appended to `growth_function_results.csv`.

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

### k = m regime (equal center and query length — thesis notation)

The gradient descent approach was used to estimate shattering capacity for $k = m \in \{1, \ldots, 9\}$ (thesis notation; since $k=m$ the swap is transparent). Data in `results2/results2/`. Results suggest approximately linear growth (roughly $2k$), though random sampling may miss specific point configurations for larger sets.

| $k = m$ | Capacity (soft-DTW) | Capacity (exact QP) |
|---------|--------------------|--------------------|
| 1 | 1 | 1 |
| 2 | 4 | 4 |
| 3 | 6 | 6 |
| 4 | 8 | 8 |
| 5 | ~9 | — |
| 6 | ~9 | — |

### Growth function ($k = m = 3$, thesis notation)

For $k = m = 3$, the growth function $\Pi(s)$ grows as $2^s$ up to $s = 6$, where it falls below $2^s$, confirming that the VC dimension lower bound is at least 6. Data in `growth_function_results.csv`.

### Comparison with the QP solver

The exact QP solver (see [`dtw_intersection`](https://github.com/JannisKre/dtw_intersection)) confirmed all $2^9 = 512$ subsets for a 9-point example ($k=7$, $m=2$ in thesis notation; code file `k2_m7.csv`), while this solver found witnesses for only 14 — illustrating the trade-off between exactness and scalability. Conversely, for a $k = m = 15$ separation task, the gradient descent approach finds a valid separator while the QP times out.

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
