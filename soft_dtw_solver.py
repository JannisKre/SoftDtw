"""
soft_dtw_solver.py
==================
Soft-DTW ball solver for shattering-capacity estimation.

Core components
---------------
- SoftDTW          : differentiable DTW distance (PyTorch, autograd-compatible)
- optimize_ball    : gradient-descent solver for center P and radius Delta
- check_shattering : tests whether a point set is shattered by Soft-DTW balls
- sequential_capacity_estimation : greedy search for the largest shattered set
- save_witnesses_csv : persist witnesses to disk

Usage (command line)
--------------------
    python soft_dtw_solver.py --m 2 --k 2 --gamma 0.1 --num_runs 3 \
        --out witnesses.csv
"""

import argparse
import contextlib
import csv
import io
import itertools
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Soft-DTW
# =============================================================================

def soft_min(a: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Soft-minimum via the LogSumExp trick.

    Parameters
    ----------
    a     : 1-D tensor of candidate values.
    gamma : smoothing parameter (> 0).

    Returns
    -------
    Scalar tensor: -gamma * log( sum_i exp(-a_i / gamma) )
    """
    return -gamma * torch.logsumexp(-a / gamma, dim=0)


class SoftDTW(nn.Module):
    """
    Differentiable Soft-DTW distance between two time series.

    Parameters
    ----------
    gamma : smoothing parameter (> 0).  Smaller values approach hard DTW.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (seq_len_x,) or (seq_len_x, dim)
        y : (seq_len_y,) or (seq_len_y, dim)

        Returns
        -------
        Scalar tensor — the Soft-DTW distance.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)

        n = x.shape[0]
        m = y.shape[0]

        # Pairwise squared Euclidean distances  (n, m)
        dist_matrix = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=2)

        INF = 1e10
        dp = [[torch.tensor(INF) for _ in range(m + 1)] for _ in range(n + 1)]
        dp[0][0] = torch.tensor(0.0)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i - 1, j - 1]
                neighbors = torch.stack([dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]])
                dp[i][j] = cost + soft_min(neighbors, self.gamma)

        return dp[n][m]


# =============================================================================
# Ball optimiser
# =============================================================================

def optimize_ball(
    Qs,
    I,
    k: int,
    gamma: float = 1.0,
    lr: float = 0.1,
    epochs: int = 500,
    margin: float = 0.1,
    verbose: bool = True,
    init_P=None,
):
    """
    Gradient-descent search for a Soft-DTW ball center P and radius Delta that
    separates the subset I from its complement.

    Parameters
    ----------
    Qs     : list of array-like time series.
    I      : indices of series that must lie *inside* the ball.
    k      : length of the center series P.
    gamma  : Soft-DTW smoothing parameter.
    lr     : Adam learning rate.
    epochs : number of gradient steps.
    margin : hinge margin added to each constraint.
    verbose: if True, print progress every 50 epochs.
    init_P : optional initial center (array-like, length k).  If None, random.

    Returns
    -------
    P     : optimised center (detached tensor, shape (k,))
    Delta : optimised radius (detached scalar tensor)
    losses: list of per-epoch loss values
    """
    Qs = [
        q.clone().detach().float() if isinstance(q, torch.Tensor)
        else torch.tensor(q, dtype=torch.float32)
        for q in Qs
    ]

    if init_P is not None:
        p0 = init_P.clone().detach().float() if isinstance(init_P, torch.Tensor) \
             else torch.tensor(init_P, dtype=torch.float32)
        p0 = p0.reshape(k)
        P = nn.Parameter(p0)
    else:
        P = nn.Parameter(torch.randn(k))
    Delta = nn.Parameter(torch.tensor(1.0))
    optimizer = optim.Adam([P, Delta], lr=lr)
    sdtw = SoftDTW(gamma=gamma)

    I_set  = set(I)
    not_I  = list(set(range(len(Qs))) - I_set)
    I_list = list(I_set)

    if verbose:
        print(f"Optimising  P (len={k}), Delta | In={I_list}  Out={not_I}")

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        dist_in  = [sdtw(Qs[i], P) for i in I_list]
        dist_out = [sdtw(Qs[j], P) for j in not_I]

        loss = (
            sum(torch.relu(d - Delta + margin) for d in dist_in) +
            sum(torch.relu(Delta - d + margin) for d in dist_out)
        )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and epoch % 50 == 0:
            avg_in  = torch.tensor([d.item() for d in dist_in]).mean().item()  if dist_in  else 0.0
            avg_out = torch.tensor([d.item() for d in dist_out]).mean().item() if dist_out else 0.0
            print(
                f"  Epoch {epoch:4d}: loss={loss.item():.4f}  "
                f"Delta={Delta.item():.4f}  "
                f"avg_in={avg_in:.4f}  avg_out={avg_out:.4f}"
            )

    return P.detach(), Delta.detach(), losses


# =============================================================================
# Hard-DTW validation
# =============================================================================

def to_1d_numpy(curve) -> np.ndarray:
    """Convert a tensor or array-like to a flat 1-D NumPy array."""
    if isinstance(curve, torch.Tensor):
        arr = curve.detach().cpu().numpy()
    else:
        arr = np.asarray(curve)
    return arr.reshape(-1)


def hard_dtw_distance(x, y) -> float:
    """Classic (hard) DTW with squared Euclidean local cost."""
    x = to_1d_numpy(x)
    y = to_1d_numpy(y)
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def validate_witness_hard_dtw(P, Qs, I):
    """
    Check strict hard-DTW separation:
        max_{i in I} DTW(P, Q_i)  <  min_{j not in I} DTW(P, Q_j)

    Returns
    -------
    (is_valid, max_in, min_out, all_distances)
    """
    n     = len(Qs)
    I_set = set(I)
    not_I = [j for j in range(n) if j not in I_set]

    distances = [hard_dtw_distance(P, Qs[t]) for t in range(n)]

    if len(I_set) == 0 or len(not_I) == 0:
        return True, math.nan, math.nan, distances

    max_in  = max(distances[i] for i in I_set)
    min_out = min(distances[j] for j in not_I)
    return max_in < min_out, float(max_in), float(min_out), distances


# =============================================================================
# Robust ball optimiser (with retries)
# =============================================================================

def _soft_dtw_separation(P_opt, Qs, I, gamma: float):
    """
    Compute actual soft-DTW separation for a candidate center P_opt.

    Returns (soft_separated, max_in_soft, min_out_soft).
    Trivial cases (empty I or empty complement) are treated as separated.
    """
    sdtw   = SoftDTW(gamma=gamma)
    I_set  = set(I)
    not_I  = [j for j in range(len(Qs)) if j not in I_set]
    I_list = list(I_set)

    if not I_list or not not_I:
        return True, math.nan, math.nan

    Qs_t = [
        q.clone().detach().float() if isinstance(q, torch.Tensor)
        else torch.tensor(q, dtype=torch.float32)
        for q in Qs
    ]
    P_t = P_opt.float() if isinstance(P_opt, torch.Tensor) else torch.tensor(P_opt, dtype=torch.float32)

    with torch.no_grad():
        soft_dists = [sdtw(q, P_t).item() for q in Qs_t]

    max_in_soft  = max(soft_dists[i] for i in I_list)
    min_out_soft = min(soft_dists[j] for j in not_I)
    return max_in_soft < min_out_soft, max_in_soft, min_out_soft


def _smart_init_P(Qs, I, k: int) -> torch.Tensor:
    """
    Heuristic initial center of length *k* for subset *I*.

    For non-empty I: linearly interpolate the mean of the in-set queries from
    their original length up/down to length k.  This places P in the same
    numerical range as the data and gives Adam a sensible starting direction.
    For empty I (or trivial cases): fall back to random.
    """
    I_list = [i for i in I if i < len(Qs)]
    if not I_list:
        return torch.randn(k)

    in_series = np.stack(
        [to_1d_numpy(Qs[i]) for i in I_list], axis=0
    )  # shape: (|I|, query_len)
    mean_series = in_series.mean(axis=0)  # shape: (query_len,)

    if len(mean_series) == k:
        return torch.tensor(mean_series, dtype=torch.float32)

    x_old = np.linspace(0, 1, len(mean_series))
    x_new = np.linspace(0, 1, k)
    resampled = np.interp(x_new, x_old, mean_series)
    return torch.tensor(resampled, dtype=torch.float32)


def optimize_ball_robust(
    Qs,
    I,
    k: int,
    gamma: float = 1.0,
    lr: float = 0.1,
    epochs: int = 500,
    loss_threshold: float = 1e-4,
    retries: int = 3,
    require_hard_dtw_validation: bool = True,
    verbose: bool = False,
    init_P=None,
):
    """
    Repeatedly calls :func:`optimize_ball` and accepts the result as soon as
    the optimised P achieves genuine soft-DTW separation
    (``max_in_soft < min_out_soft``), regardless of the loss value.
    The ``loss_threshold`` is kept as a fast-accept shortcut: if the loss
    already dropped below it we skip the extra forward pass.

    On the first retry, ``init_P`` is used as the initial center (warm start).
    Subsequent retries always use a fresh random initialisation.

    Returns
    -------
    (success, P, Delta, max_in_dtw, min_out_dtw, hard_valid)
    """
    def _try_accept(P_candidate, Delta_candidate):
        """Return (success, P, Delta, max_in, min_out, hard_valid) if P_candidate separates."""
        sep, max_in_s, min_out_s = _soft_dtw_separation(P_candidate, Qs, I, gamma)
        if not sep:
            return None
        if require_hard_dtw_validation:
            hard_valid, max_in, min_out, _ = validate_witness_hard_dtw(P_candidate, Qs, I)
            if hard_valid:
                return True, P_candidate, Delta_candidate, max_in, min_out, True
            return None
        return True, P_candidate, Delta_candidate, max_in_s, min_out_s, False

    # ── Step 0: check warm start before any optimisation ─────────────────────
    if init_P is not None:
        p0 = init_P.clone().detach().float() if isinstance(init_P, torch.Tensor) \
             else torch.tensor(init_P, dtype=torch.float32)
        p0 = p0.reshape(k)
        # Compute a sensible Delta from the warm-start distances
        sdtw_tmp = SoftDTW(gamma=gamma)
        I_set_tmp = set(I)
        not_I_tmp = [j for j in range(len(Qs)) if j not in I_set_tmp]
        if I_set_tmp and not_I_tmp:
            with torch.no_grad():
                Qs_t = [q.clone().detach().float() if isinstance(q, torch.Tensor)
                        else torch.tensor(q, dtype=torch.float32) for q in Qs]
                ds = [sdtw_tmp(Qs_t[j], p0).item() for j in range(len(Qs))]
            mid = (max(ds[i] for i in I_set_tmp) + min(ds[j] for j in not_I_tmp)) / 2.0
            delta0 = torch.tensor(mid)
        else:
            delta0 = torch.tensor(1.0)
        result = _try_accept(p0, delta0)
        if result is not None:
            return result

    # ── Step 1: gradient-based retries ───────────────────────────────────────
    for attempt in range(retries):
        if attempt == 0:
            # First attempt: use provided init_P, smart heuristic, or random
            current_init = init_P if init_P is not None else _smart_init_P(Qs, I, k)
        else:
            current_init = None  # subsequent retries: pure random restart
        if verbose:
            P_opt, Delta_opt, losses = optimize_ball(Qs, I, k, gamma, lr, epochs, init_P=current_init)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                P_opt, Delta_opt, losses = optimize_ball(Qs, I, k, gamma, lr, epochs, init_P=current_init)

        # Fast-accept if loss is tiny
        if losses[-1] < loss_threshold:
            result = _try_accept(P_opt, Delta_opt)
            if result is not None:
                return result

        # Main criterion: actual soft-DTW separation
        result = _try_accept(P_opt, Delta_opt)
        if result is not None:
            return result

    return False, None, None, math.nan, math.nan, False


# =============================================================================
# Shattering check
# =============================================================================

class ProjectedRuntimeExceeded(RuntimeError):
    """Raised when a shattering check is projected to exceed a time budget."""

def check_shattering(Qs, k: int, gamma: float = 1.0, epochs: int = 500, retries: int = 5,
                     verbose: bool = False, validation: bool = True, init_witnesses=None,
                     max_projected_total_seconds: float | None = None):
    """
    Test whether the set *Qs* is shattered by Soft-DTW balls of center
    complexity *k*.

    Parameters
    ----------
    init_witnesses : optional dict { tuple(int,...) -> {'P': array} }
        Warm-start initialisation for each subset.  If a subset key is present
        its stored P is used as the first-retry initialisation for Adam.
        Typical use: pass the ``witnesses_csv`` dict from
        :func:`load_point_set_from_csv` to seed the solver from hard-DTW
        witnesses.
    max_projected_total_seconds : optional float
        Abort the shattering check when one subset runtime already projects
        the full ``2^|Qs|`` subset pass above this time budget.

    Returns
    -------
    is_shattered : bool
    witnesses    : dict  { tuple(subset_I) -> witness_info_dict }
    """
    indices  = list(range(len(Qs)))
    witnesses = {}
    init_w = init_witnesses or {}
    total_subsets = 2 ** len(Qs)

    for r in range(len(Qs) + 1):
        for subset_I in itertools.combinations(indices, r):
            I_list = list(subset_I)

            # Use stored witness P as warm start if available
            hint = init_w.get(tuple(I_list))
            init_P = hint["P"] if hint is not None else None

            subset_start = time.perf_counter()

            success, P_opt, Delta_opt, max_in_dtw, min_out_dtw, hard_valid = optimize_ball_robust(
                Qs,
                I_list,
                k,
                gamma=gamma,
                epochs=epochs,
                retries=retries,
                require_hard_dtw_validation=validation,
                verbose=verbose,
                init_P=init_P,
            )

            subset_elapsed = time.perf_counter() - subset_start
            if (
                max_projected_total_seconds is not None
                and subset_elapsed * total_subsets > max_projected_total_seconds
            ):
                raise ProjectedRuntimeExceeded(
                    "Projected full shattering check would take about "
                    f"{subset_elapsed * total_subsets / 3600.0:.2f}h "
                    f"({total_subsets} subsets × {subset_elapsed:.2f}s), "
                    f"which exceeds the limit of {max_projected_total_seconds / 3600.0:.2f}h."
                )

            if not success:
                if verbose:
                    print(f"  Failed subset: {I_list}")
                return False, witnesses

            witnesses[tuple(I_list)] = {
                "P":            to_1d_numpy(P_opt),
                "Delta":        float(Delta_opt.item() if isinstance(Delta_opt, torch.Tensor) else Delta_opt),
                "max_in_dtw":   max_in_dtw,
                "min_out_dtw":  min_out_dtw,
                "hard_dtw_valid": bool(hard_valid),
            }

    return True, witnesses


# =============================================================================
# CSV persistence
# =============================================================================

def save_witnesses_csv(shattered_set, witnesses, out_csv_path: str, verbose: bool = False) -> None:
    """
    Write the shattered set and all witness balls to a CSV file.

    Columns
    -------
    row_type, point_index, subset_I, Q_values, P_values, Delta,
    hard_dtw_max_in, hard_dtw_min_out, hard_dtw_valid
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_type", "point_index", "subset_I", "Q_values",
            "P_values", "Delta", "hard_dtw_max_in", "hard_dtw_min_out", "hard_dtw_valid",
        ])

        writer.writerow(["SHATTERED_SET", "", "", "", "", "", "", "", ""])
        for idx, q in enumerate(shattered_set):
            q_vals = ";".join(f"{v:.10g}" for v in to_1d_numpy(q))
            writer.writerow(["POINT", idx, "", q_vals, "", "", "", "", ""])

        writer.writerow(["WITNESSES", "", "", "", "", "", "", "", ""])
        for subset_key in sorted(witnesses.keys(), key=lambda t: (len(t), t)):
            w          = witnesses[subset_key]
            subset_str = "{" + ",".join(map(str, subset_key)) + "}"
            p_vals     = ";".join(f"{v:.10g}" for v in w["P"])
            writer.writerow([
                "SUBSET_WITNESS", "",
                subset_str, "",
                p_vals,
                f"{w['Delta']:.10g}",
                "" if math.isnan(w["max_in_dtw"])  else f"{w['max_in_dtw']:.10g}",
                "" if math.isnan(w["min_out_dtw"]) else f"{w['min_out_dtw']:.10g}",
                int(bool(w["hard_dtw_valid"])),
            ])

    if verbose:
        print(f"Witnesses saved to: {out_csv_path}")


# =============================================================================
# Sequential capacity estimation
# =============================================================================

def _with_run_suffix(path: str, run_idx: int, num_runs: int) -> str:
    """Append '_run<N>' to *path* when more than one run is requested."""
    if num_runs <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_run{run_idx}{ext or '.csv'}"


def _single_sequential_run(
    m: int,
    k: int,
    gamma: float,
    max_retries_step4: int,
    epochs: int,
    retries: int,
    max_d: int,
    witness_csv_path: str,
    run_idx: int,
    num_runs: int,
    verbose: bool,
    validation: bool = True,
    max_projected_shattering_seconds: float | None = None,
):
    """Execute one greedy sequential run and return (d_max, X, witnesses)."""
    d                = 0
    X                = []
    witnesses_final  = {}

    while d < max_d:
        print(f"Run {run_idx}/{num_runs} | current d={d}")
        success = False

        for _ in range(max_retries_step4 + 1):
            new_Q      = torch.randn(m, 1)
            current_X  = X + [new_Q]
            try:
                is_shattered, current_witnesses = check_shattering(
                    current_X,
                    k,
                    gamma,
                    epochs=epochs,
                    retries=retries,
                    verbose=verbose,
                    validation=validation,
                    max_projected_total_seconds=max_projected_shattering_seconds,
                )
            except ProjectedRuntimeExceeded as exc:
                print(f"Run {run_idx}/{num_runs} | stopping at d={d + 1}: {exc}")
                success = False
                break

            if is_shattered:
                X               = current_X
                witnesses_final = current_witnesses
                d              += 1
                success         = True
                break

        if not success:
            break

    out_path = _with_run_suffix(witness_csv_path, run_idx, num_runs)
    save_witnesses_csv(X, witnesses_final, out_path, verbose=verbose)
    return d, X, witnesses_final


def sequential_capacity_estimation(
    m: int,
    k: int,
    gamma: float = 1.0,
    max_retries_step4: int = 5,
    epochs: int = 500,
    retries: int = 5,
    max_d: int = 15,
    witness_csv_path: str = "witnesses_sequential_capacity.csv",
    num_runs: int = 1,
    verbose: bool = False,
    validation: bool = True,
    max_projected_shattering_seconds: float | None = None,
):
    """
    Estimate the shattering capacity for query series in R^m and center in R^k.

    Parameters
    ----------
    m                 : dimensionality / length of each query series.
    k                 : length of the ball center P.
    gamma             : Soft-DTW smoothing parameter.
    max_retries_step4 : maximum attempts to add a new point before giving up.
    epochs            : gradient steps per subset-separation optimisation.
    retries           : random restarts per subset-separation optimisation.
    max_d             : upper bound on d to search for.
    witness_csv_path  : output CSV path (suffix '_run<N>' added for multi-run).
    num_runs          : number of independent greedy runs.
    verbose           : detailed per-subset output when True.
    validation        : whether to require hard-DTW validation of witnesses.
    max_projected_shattering_seconds : optional float
        Abort a candidate shattering test when one subset runtime already
        projects the full ``2^d`` subset pass above this time budget.

    Returns
    -------
    num_runs == 1 : (d_max, X_final, witnesses_final)
    num_runs  > 1 : (best_d, best_X, best_witnesses, all_d_values)
    """
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    all_results   = []
    all_d_values  = []

    for run_idx in range(1, num_runs + 1):
        d, X, witnesses = _single_sequential_run(
            m=m, k=k, gamma=gamma,
            max_retries_step4=max_retries_step4,
            epochs=epochs,
            retries=retries,
            max_d=max_d,
            witness_csv_path=witness_csv_path,
            run_idx=run_idx,
            num_runs=num_runs,
            verbose=verbose,
            validation=validation,
            max_projected_shattering_seconds=max_projected_shattering_seconds,
        )
        all_results.append((d, X, witnesses))
        all_d_values.append(d)
        print(f"Run {run_idx}/{num_runs} finished with d_max={d}")

    if num_runs == 1:
        return all_results[0]

    best_idx = int(np.argmax(all_d_values))
    best_d, best_X, best_witnesses = all_results[best_idx]
    print(f"Best over {num_runs} runs: d_max={best_d}")
    return best_d, best_X, best_witnesses, all_d_values


# =============================================================================
# Load / validate witnesses from the compact k2_m7.csv format
# =============================================================================

def load_point_set_from_csv(csv_path: str):
    """
    Parse the compact experiment CSV (columns: k, m, run_id, d_max,
    points_json, subset, witness_json, delta).

    CSV convention
    --------------
    - ``k``  column = query series length  (solver's ``m``)
    - ``m``  column = center series length (solver's ``k``)
    - subset indices are **1-based**; this function converts them to 0-based.
    - The row with ``subset == "__POINTS__"`` carries the full point set.

    Returns
    -------
    points         : list[torch.Tensor]  – the query series
    query_len      : int  – length of each query  (CSV ``k``)
    center_len     : int  – length of each center (CSV ``m``)
    witnesses_csv  : dict { tuple(int, ...) -> {'P': np.ndarray, 'Delta': float} }
    """
    import json

    points: list = []
    query_len  = None
    center_len = None
    witnesses_csv: dict = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if query_len is None:
                query_len  = int(row["k"])   # query length  (CSV k = solver m)
                center_len = int(row["m"])   # center length (CSV m = solver k)

            if row["subset"] == "__POINTS__":
                raw = json.loads(row["points_json"])
                points = [torch.tensor(p, dtype=torch.float32) for p in raw]
            elif row.get("witness_json", "").strip():
                # Convert 1-based CSV indices to 0-based
                indices = tuple(int(x) - 1 for x in row["subset"].split(","))
                P_arr   = np.array(json.loads(row["witness_json"]), dtype=np.float64)
                delta   = float(row["delta"])
                witnesses_csv[indices] = {"P": P_arr, "Delta": delta}

    return points, query_len, center_len, witnesses_csv


def validate_csv_witnesses(points, witnesses_csv, verbose: bool = True):
    """
    Check every stored witness with hard DTW strict separation:
        max_{i in I} DTW(P, Q_i)  <  min_{j not in I} DTW(P, Q_j)

    Prints a summary and returns a dict of per-subset results.
    """
    n_total  = len(witnesses_csv)
    n_pass   = 0
    failures = []

    results = {}
    for subset_key, w in sorted(witnesses_csv.items(), key=lambda t: (len(t[0]), t[0])):
        I_list = list(subset_key)
        valid, max_in, min_out, _ = validate_witness_hard_dtw(w["P"], points, I_list)
        results[subset_key] = {
            "valid":   valid,
            "max_in":  max_in,
            "min_out": min_out,
            "Delta":   w["Delta"],
        }
        if valid:
            n_pass += 1
        else:
            failures.append((subset_key, max_in, min_out))

    if verbose:
        print(f"\nHard-DTW witness validation: {n_pass}/{n_total} passed")
        if failures:
            print("Failed subsets:")
            for key, mi, mo in failures:
                idx_str = "{" + ",".join(str(i) for i in key) + "}"
                print(f"  I={idx_str}  max_in={mi:.6g}  min_out={mo:.6g}")
        else:
            print("All witnesses valid under hard DTW.")

    return results


def test_shattering_from_csv(
    csv_path: str,
    gamma: float = 0.1,
    epochs: int = 300,
    retries: int = 3,
    validation: bool = True,
    verbose: bool = False,
):
    """
    Load the fixed point set from *csv_path*, validate the stored hard-DTW
    witnesses, then run the Soft-DTW solver to independently verify shattering.

    Parameters
    ----------
    csv_path   : path to the compact CSV file (k2_m7.csv format)
    gamma      : Soft-DTW smoothing parameter
    epochs     : gradient steps per optimisation call
    retries    : retries per subset in :func:`optimize_ball_robust`
    validation : whether to require hard-DTW validation for Soft-DTW witnesses
    verbose    : detailed per-subset output during Soft-DTW optimisation

    Returns
    -------
    (is_shattered_softdtw, softdtw_witnesses, validation_results)
    """
    points, query_len, center_len, witnesses_csv = load_point_set_from_csv(csv_path)

    print(f"Loaded {len(points)} points from '{csv_path}'")
    print(f"  Query length  (solver m) = {query_len}")
    print(f"  Center length (solver k) = {center_len}")
    print(f"  Stored witnesses         = {len(witnesses_csv)}")
    print(f"  Expected subsets (2^d)   = {2 ** len(points)}")

    # ── 1. Validate stored witnesses under hard DTW ───────────────────────────
    print("\n── Step 1: Validate stored witnesses (hard DTW) ─────────────────")
    val_results = validate_csv_witnesses(points, witnesses_csv, verbose=True)
    n_valid = sum(1 for r in val_results.values() if r["valid"])
    print(f"Hard-DTW validation: {n_valid}/{len(val_results)} witnesses pass strict separation.")

    # ── 2. Run Soft-DTW solver from scratch ───────────────────────────────────
    print(f"\n── Step 2: Soft-DTW shattering check (gamma={gamma}, k={center_len}) ─")
    is_shattered, softdtw_witnesses = check_shattering(
        points,
        k=center_len,
        gamma=gamma,
        epochs=epochs,
        retries=retries,
        verbose=verbose,
        validation=validation,
    )

    n_found = len(softdtw_witnesses)
    n_total = 2 ** len(points)
    print(f"\nSoft-DTW result  : {'SHATTERED ✓' if is_shattered else 'NOT shattered ✗'}")
    print(f"Subsets solved   : {n_found} / {n_total}")

    return is_shattered, softdtw_witnesses, val_results


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Soft-DTW ball shattering-capacity estimator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode selection ────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--test_csv",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help=(
            "Load a fixed point set from this CSV file (k2_m7 format), "
            "validate its stored witnesses with hard DTW, then run the "
            "Soft-DTW solver to independently verify shattering."
        ),
    )

    # ── Sequential capacity estimation options ────────────────────────────────
    parser.add_argument("--m",        type=int,   default=2,    help="Query series length")
    parser.add_argument("--k",        type=int,   default=2,    help="Center series length")
    parser.add_argument("--num_runs", type=int,   default=3,    help="Number of independent runs")
    parser.add_argument("--max_d",    type=int,   default=15,   help="Maximum d to search")
    parser.add_argument("--retries",  type=int,   default=3,    help="Retries per new point / subset")
    parser.add_argument("--out",      type=str,   default="witnesses.csv", help="Output CSV path")

    # ── Shared options ────────────────────────────────────────────────────────
    parser.add_argument("--gamma",         type=float, default=0.1,  help="Soft-DTW smoothing parameter")
    parser.add_argument("--epochs",        type=int,   default=300,  help="Gradient steps per optimisation")
    parser.add_argument("--no_validation", action="store_true",      help="Skip hard-DTW validation")
    parser.add_argument("--verbose",       action="store_true",      help="Detailed output")
    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Test a fixed point set loaded from CSV ────────────────────────────────
    if args.test_csv:
        test_shattering_from_csv(
            csv_path=args.test_csv,
            gamma=args.gamma,
            epochs=args.epochs,
            retries=args.retries,
            validation=not args.no_validation,
            verbose=args.verbose,
        )
        return

    # ── Sequential capacity estimation ───────────────────────────────────────
    result = sequential_capacity_estimation(
        m=args.m,
        k=args.k,
        gamma=args.gamma,
        max_retries_step4=args.retries,
        max_d=args.max_d,
        witness_csv_path=args.out,
        num_runs=args.num_runs,
        verbose=args.verbose,
        validation=not args.no_validation,
    )

    if len(result) == 3:
        d_max, _, witnesses = result
        print(f"\nEstimated capacity : {d_max}")
        print(f"Witnesses stored   : {len(witnesses)}")
    else:
        best_d, _, best_witnesses, all_ds = result
        print(f"\nRun d_max values   : {all_ds}")
        print(f"Best capacity      : {best_d}")
        print(f"Witnesses stored   : {len(best_witnesses)}")


if __name__ == "__main__":
    main()
