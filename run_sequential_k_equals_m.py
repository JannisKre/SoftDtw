"""
run_sequential_k_equals_m.py
=============================
Server-side script: run sequential shattering capacity estimation for k = m.

For each value of m specified via --m_values, sets k = m and calls
sequential_capacity_estimation with hard-DTW validation enabled.  Results
are written to per-(k,m) CSV files and a summary CSV.

Usage examples
--------------
    # Single value, 3 runs, output to results/
    python run_sequential_k_equals_m.py --m_values 2 --num_runs 3 --out_dir results/

    # Sweep m = 1..5, 1 run each, verbose
    python run_sequential_k_equals_m.py --m_values 1 2 3 4 5 --num_runs 1 --verbose

    # High-budget run: m = 1..10, 5 runs each, more retries, bigger max_d
    python run_sequential_k_equals_m.py \\
        --m_values 1 2 3 4 5 6 7 8 9 10 \\
        --num_runs 5 --max_d 30 --retries 10 --epochs 1000 \\
        --gamma 1.0 --out_dir results/

Outputs
-------
  <out_dir>/witnesses_k{k}_m{m}[_run{n}].csv   per-run witnesses
  <out_dir>/summary.csv                         aggregated results
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from soft_dtw_solver import sequential_capacity_estimation


# =============================================================================
# Argument parsing
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Sequential shattering capacity sweep with k = m",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m_values", type=int, nargs="+", default=[1, 2, 3],
        metavar="M",
        help="List of m values to sweep (k is set equal to m for each).",
    )
    p.add_argument("--gamma",    type=float, default=0.1,  help="Soft-DTW smoothing parameter.")
    p.add_argument("--epochs",   type=int,   default=500,  help="Gradient steps per optimisation call.")
    p.add_argument("--retries",  type=int,   default=5,    help="Retries per subset / new-point attempt.")
    p.add_argument("--max_d",    type=int,   default=20,   help="Upper bound on shattering dimension.")
    p.add_argument("--num_runs", type=int,   default=1,    help="Independent greedy runs per (k,m) pair.")
    p.add_argument(
        "--sampling_mode",
        type=str,
        choices=["gaussian", "near_unit_sphere"],
        default="near_unit_sphere",
        help="How new query series are sampled during the greedy search.",
    )
    p.add_argument(
        "--sampling_radius_noise",
        type=float,
        default=0.05,
        help="Radial noise used when --sampling_mode near_unit_sphere.",
    )
    p.add_argument(
        "--max_projected_shattering_seconds",
        type=float,
        default=86400.0,
        help="Abort a candidate d when one subset runtime projects the full 2^d subset pass beyond this many seconds.",
    )
    p.add_argument(
        "--out_dir", type=str, default=".",
        help="Directory for witness CSVs and summary.csv.",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-subset optimisation detail.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_rows = []

    print("=" * 65)
    print("  Sequential shattering capacity — k = m sweep")
    print("=" * 65)
    print(f"  m values   : {args.m_values}")
    print(f"  gamma      : {args.gamma}")
    print(f"  epochs     : {args.epochs}")
    print(f"  retries    : {args.retries}")
    print(f"  max_d      : {args.max_d}")
    print(f"  num_runs   : {args.num_runs}")
    print(f"  sampling   : {args.sampling_mode} (noise={args.sampling_radius_noise})")
    print(f"  time guard : {args.max_projected_shattering_seconds}s")
    print(f"  output dir : {os.path.abspath(args.out_dir)}")
    print("=" * 65)

    for m in args.m_values:
        k = m
        out_csv = os.path.join(args.out_dir, f"witnesses_k{k}_m{m}.csv")

        print(f"\n[m={m}, k={k}]  starting ...")
        t0 = time.time()

        result = sequential_capacity_estimation(
            m=m,
            k=k,
            gamma=args.gamma,
            max_retries_step4=args.retries,
            epochs=args.epochs,
            retries=args.retries,
            max_d=args.max_d,
            witness_csv_path=out_csv,
            num_runs=args.num_runs,
            verbose=args.verbose,
            validation=True,          # always use hard-DTW validation
            max_projected_shattering_seconds=args.max_projected_shattering_seconds,
            sampling_mode=args.sampling_mode,
            sampling_radius_noise=args.sampling_radius_noise,
        )

        elapsed = time.time() - t0

        if args.num_runs == 1:
            d_max, _, witnesses = result
            all_d = [d_max]
        else:
            d_max, _, witnesses, all_d = result

        hit_max = d_max >= args.max_d
        print(
            f"[m={m}, k={k}]  d_max={d_max}  "
            f"({'≥ MAX_D — increase --max_d for a tighter bound' if hit_max else 'converged'})"
            f"  [{elapsed:.1f}s]"
        )

        summary_rows.append({
            "m":        m,
            "k":        k,
            "d_max":    d_max,
            "d_mean":   round(float(np.mean(all_d)), 4),
            "d_min":    min(all_d),
            "all_d":    ";".join(map(str, all_d)),
            "hit_max_d": int(hit_max),
            "elapsed_s": round(elapsed, 1),
            "gamma":    args.gamma,
            "epochs":   args.epochs,
            "retries":  args.retries,
            "max_d":    args.max_d,
            "num_runs": args.num_runs,
            "sampling_mode": args.sampling_mode,
            "sampling_radius_noise": args.sampling_radius_noise,
            "max_projected_shattering_seconds": args.max_projected_shattering_seconds,
        })

    # ── Write summary CSV ─────────────────────────────────────────────────────
    fieldnames = [
        "m", "k", "d_max", "d_mean", "d_min", "all_d",
        "hit_max_d", "elapsed_s",
        "gamma", "epochs", "retries", "max_d", "num_runs",
        "sampling_mode", "sampling_radius_noise",
        "max_projected_shattering_seconds",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'=' * 65}")
    print("  Summary")
    print(f"{'=' * 65}")
    print(f"  {'m':>4}  {'k':>4}  {'d_max':>7}  {'d_mean':>7}  {'elapsed':>9}  {'note'}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*20}")
    for r in summary_rows:
        note = "≥ MAX_D" if r["hit_max_d"] else ""
        print(
            f"  {r['m']:>4}  {r['k']:>4}  {r['d_max']:>7}  "
            f"{r['d_mean']:>7}  {r['elapsed_s']:>8.1f}s  {note}"
        )
    print(f"\n  Summary saved to: {summary_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
