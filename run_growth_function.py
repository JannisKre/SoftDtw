"""
run_growth_function.py
======================
Server-side script: estimate the growth function Pi_H(s) for Soft-DTW balls
with a fixed hypothesis class (m, k = m).

For each sample size s specified via --s_values, draws --num_samples random
point sets of size s with each point in R^m, counts the realizable subsets
(without short-circuiting), and writes per-s results plus a summary CSV.

Usage examples
--------------
    # m=7, k=7, s = 1..15, 3 samples each
    python run_growth_function.py --m 7 --s_values $(seq 1 15) --num_samples 3

    # Same with a time guard of 2 h per (s, sample)
    python run_growth_function.py --m 7 --s_values $(seq 1 15) \\
        --num_samples 3 --max_projected_seconds 7200 --out_dir results_growth/

    # Lower budget quick test
    python run_growth_function.py --m 2 --s_values 1 2 3 4 5 \\
        --num_samples 5 --epochs 300 --out_dir results_growth_m2/

Outputs
-------
  <out_dir>/growth_s{s}.csv      per-s per-sample counts (one row per sample)
  <out_dir>/summary.csv          one row per s with Pi_hat, mean, std, …
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import torch

from soft_dtw_solver import (
    optimize_ball_robust,
    to_1d_numpy,
    _sample_query_series,
    ProjectedRuntimeExceeded,
)


# =============================================================================
# Core counting function
# =============================================================================

def count_realizable_subsets(
    Qs,
    k: int,
    gamma: float = 1.0,
    epochs: int = 500,
    retries: int = 5,
    validation: bool = True,
    max_projected_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[int, dict]:
    """
    Count how many of the 2^|Qs| subsets of *Qs* can be realised by a
    Soft-DTW ball with centre complexity *k*.

    Does NOT short-circuit on failure — every subset is tested.
    Raises ``ProjectedRuntimeExceeded`` if the first subset's runtime already
    projects the full 2^|Qs| pass beyond ``max_projected_seconds``.

    Returns
    -------
    n_realizable : int
    witnesses    : dict  { tuple(subset_I) -> witness_info_dict }
    """
    indices       = list(range(len(Qs)))
    witnesses     = {}
    total_subsets = 2 ** len(Qs)
    first_subset  = True

    for r in range(len(Qs) + 1):
        for subset_I in itertools.combinations(indices, r):
            I_list = list(subset_I)

            t_sub = time.perf_counter()
            success, P_opt, Delta_opt, max_in_dtw, min_out_dtw, hard_valid = optimize_ball_robust(
                Qs,
                I_list,
                k,
                gamma=gamma,
                epochs=epochs,
                retries=retries,
                require_hard_dtw_validation=validation,
                verbose=False,
            )
            sub_elapsed = time.perf_counter() - t_sub

            if first_subset and max_projected_seconds is not None:
                projected = sub_elapsed * total_subsets
                if projected > max_projected_seconds:
                    raise ProjectedRuntimeExceeded(
                        f"Projected runtime {projected / 3600:.2f} h "
                        f"({total_subsets} subsets × {sub_elapsed:.2f}s) "
                        f"exceeds limit of {max_projected_seconds / 3600:.2f} h."
                    )
            first_subset = False

            if success:
                witnesses[tuple(I_list)] = {
                    "P":              to_1d_numpy(P_opt),
                    "Delta":          float(
                        Delta_opt.item()
                        if isinstance(Delta_opt, torch.Tensor)
                        else Delta_opt
                    ),
                    "max_in_dtw":     max_in_dtw,
                    "min_out_dtw":    min_out_dtw,
                    "hard_dtw_valid": bool(hard_valid),
                }
            if verbose:
                tick = "✓" if success else "✗"
                print(f"  {tick}  I={I_list}")

    n_realizable = len(witnesses)
    if verbose:
        print(
            f"\n  → {n_realizable} / {total_subsets} realizable "
            f"({'fully shattered' if n_realizable == total_subsets else 'partial'})"
        )
    return n_realizable, witnesses


# =============================================================================
# Argument parsing
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Growth function estimation for Soft-DTW balls (fixed m, k=m).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m", type=int, default=7,
        help="Query-series length (dimensionality). Centre length k is set to m.",
    )
    p.add_argument(
        "--s_values", type=int, nargs="+", default=list(range(1, 16)),
        metavar="S",
        help="Sample sizes s to sweep (argument of the growth function).",
    )
    p.add_argument(
        "--num_samples", type=int, default=3,
        help="Independent random point sets drawn per s.",
    )
    p.add_argument("--gamma",   type=float, default=0.1,  help="Soft-DTW smoothing parameter.")
    p.add_argument("--epochs",  type=int,   default=500,  help="Gradient steps per optimisation call.")
    p.add_argument("--retries", type=int,   default=5,    help="Restarts per subset optimisation.")
    p.add_argument(
        "--sampling_mode",
        type=str,
        choices=["gaussian", "near_unit_sphere"],
        default="near_unit_sphere",
        help="Sampling distribution for query series.",
    )
    p.add_argument(
        "--sampling_radius_noise",
        type=float,
        default=0.05,
        help="Radial noise for near_unit_sphere sampling.",
    )
    p.add_argument(
        "--max_projected_seconds",
        type=float,
        default=None,
        help=(
            "Abort a (s, sample) pair when the first subset runtime projects "
            "the full 2^s pass beyond this many seconds. Default: no limit."
        ),
    )
    p.add_argument(
        "--no_validation", action="store_true",
        help="Disable hard-DTW validation (accept Soft-DTW witnesses only).",
    )
    p.add_argument(
        "--out_dir", type=str, default="results_growth",
        help="Output directory for per-s CSVs and summary.csv.",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-subset detail.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args      = _parse_args()
    m         = args.m
    k         = m           # k = m by design
    validation = not args.no_validation

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_rows = []

    print("=" * 65)
    print("  Growth function estimation — Soft-DTW balls")
    print("=" * 65)
    print(f"  m (fixed)   : {m}")
    print(f"  k (fixed)   : {k}")
    print(f"  s values    : {args.s_values}")
    print(f"  num_samples : {args.num_samples}")
    print(f"  gamma       : {args.gamma}")
    print(f"  epochs      : {args.epochs}")
    print(f"  retries     : {args.retries}")
    print(f"  sampling    : {args.sampling_mode} (noise={args.sampling_radius_noise})")
    print(f"  validation  : {'hard-DTW' if validation else 'soft-DTW only'}")
    print(f"  time guard  : {f'{args.max_projected_seconds:.0f}s' if args.max_projected_seconds else 'none'}")
    print(f"  output dir  : {os.path.abspath(args.out_dir)}")
    print("=" * 65)

    # ── Per-s fieldnames for individual sample CSV ────────────────────────────
    sample_fieldnames = [
        "s", "m", "k", "2^s",
        "sample_idx", "n_realizable", "frac", "aborted", "elapsed_s",
    ]

    for s in args.s_values:
        total_subsets = 2 ** s
        per_s_csv     = os.path.join(args.out_dir, f"growth_s{s}.csv")

        print(f"\n{'─'*65}")
        print(f"  s = {s}   ({total_subsets} subsets per sample)")
        print(f"{'─'*65}")

        counts    = []   # int or None (aborted)
        t_start_s = time.time()

        with open(per_s_csv, "w", newline="") as f_sample:
            writer_sample = csv.DictWriter(f_sample, fieldnames=sample_fieldnames)
            writer_sample.writeheader()

            for sample_idx in range(args.num_samples):
                Qs = [
                    _sample_query_series(
                        m,
                        sampling_mode=args.sampling_mode,
                        sampling_radius_noise=args.sampling_radius_noise,
                    )
                    for _ in range(s)
                ]

                t0 = time.time()
                aborted = False
                n_realizable = None
                try:
                    n_realizable, _ = count_realizable_subsets(
                        Qs, k,
                        gamma=args.gamma,
                        epochs=args.epochs,
                        retries=args.retries,
                        validation=validation,
                        max_projected_seconds=args.max_projected_seconds,
                        verbose=args.verbose,
                    )
                except ProjectedRuntimeExceeded as exc:
                    aborted = True
                    print(f"  sample {sample_idx + 1:2d}/{args.num_samples} : "
                          f"ABORTED — {exc}")

                elapsed = time.time() - t0
                counts.append(n_realizable)

                if not aborted:
                    frac = n_realizable / total_subsets
                    print(
                        f"  sample {sample_idx + 1:2d}/{args.num_samples} : "
                        f"{n_realizable:6d} / {total_subsets}  "
                        f"({100 * frac:.1f}%)  [{elapsed:.1f}s]"
                    )
                else:
                    frac = None

                writer_sample.writerow({
                    "s":           s,
                    "m":           m,
                    "k":           k,
                    "2^s":         total_subsets,
                    "sample_idx":  sample_idx + 1,
                    "n_realizable": n_realizable if n_realizable is not None else "",
                    "frac":        f"{frac:.6f}" if frac is not None else "",
                    "aborted":     int(aborted),
                    "elapsed_s":   round(elapsed, 2),
                })

        elapsed_s    = time.time() - t_start_s
        valid_counts = [c for c in counts if c is not None]
        pi_hat       = int(max(valid_counts)) if valid_counts else None
        n_aborted    = counts.count(None)

        status = (
            f"Pi_hat = {pi_hat} / {total_subsets}  "
            f"({'FULLY SHATTERED ✓' if pi_hat == total_subsets else 'partial'})"
            if pi_hat is not None
            else "all aborted"
        )
        print(
            f"  → {status}   "
            f"[aborted: {n_aborted}/{args.num_samples}]   "
            f"total {elapsed_s:.0f}s"
        )

        summary_rows.append({
            "s":                   s,
            "m":                   m,
            "k":                   k,
            "2^s":                 total_subsets,
            "pi_hat":              pi_hat if pi_hat is not None else "",
            "mean_count":          round(float(np.mean(valid_counts)), 4) if valid_counts else "",
            "std_count":           round(float(np.std(valid_counts)),  4) if valid_counts else "",
            "min_count":           int(min(valid_counts)) if valid_counts else "",
            "frac_max":            round(pi_hat / total_subsets, 6) if pi_hat is not None else "",
            "fully_shattered":     int(pi_hat == total_subsets) if pi_hat is not None else "",
            "n_aborted":           n_aborted,
            "elapsed_s":           round(elapsed_s, 1),
            "gamma":               args.gamma,
            "epochs":              args.epochs,
            "retries":             args.retries,
            "sampling_mode":       args.sampling_mode,
            "sampling_radius_noise": args.sampling_radius_noise,
            "num_samples":         args.num_samples,
            "max_projected_seconds": args.max_projected_seconds if args.max_projected_seconds else "",
            "validation":          int(validation),
        })

    # ── Write summary CSV ─────────────────────────────────────────────────────
    fieldnames = [
        "s", "m", "k", "2^s",
        "pi_hat", "mean_count", "std_count", "min_count",
        "frac_max", "fully_shattered", "n_aborted", "elapsed_s",
        "gamma", "epochs", "retries",
        "sampling_mode", "sampling_radius_noise",
        "num_samples", "max_projected_seconds", "validation",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*65}")
    print("  Summary")
    print(f"{'='*65}")
    print(f"  {'s':>3}  {'2^s':>6}  {'Pi_hat':>8}  {'coverage':>9}  {'aborted':>7}")
    print(f"  {'─'*3}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*7}")
    for r in summary_rows:
        cov = (
            f"{100 * float(r['frac_max']):7.1f}%"
            if r["frac_max"] != "" else "      —"
        )
        print(
            f"  {r['s']:>3}  {r['2^s']:>6}  "
            f"{str(r['pi_hat']):>8}  {cov}  "
            f"{r['n_aborted']:>2}/{r['num_samples']:>2}"
        )
    print(f"\n  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
