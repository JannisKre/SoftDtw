"""
benchmark_separator_runtime.py
==============================
Standardized runtime benchmarks for separator solvers over a grid of query
lengths ``m`` and center lengths ``k``.

The script generates the same random benchmark instances for every compared
method, measures wall-clock runtime with ``time.perf_counter()``, and writes
both detailed and aggregated CSV files.

Built-in method
---------------
- ``softdtw``: wraps ``optimize_ball_robust`` from ``soft_dtw_solver.py``

Optional external method
------------------------
You can compare against another solver by passing ``--external_solver`` as
``module:function``. The external callable should accept the signature

    solver(Qs, I, k, **kwargs)

and either return
- a boolean ``success`` value, or
- a tuple/list whose first element is ``success``.

Example
-------
    python benchmark_separator_runtime.py \
        --m_values 4 8 12 \
        --k_values 4 8 12 \
        --num_instances 10 \
        --methods softdtw qp \
        --external_solver my_qp_solver:solve_separator \
        --external_solver_kwargs '{"epsilon": 10.0}' \
        --out_dir benchmark_results
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from soft_dtw_solver import optimize_ball_robust


@dataclass
class BenchmarkInstance:
    m: int
    k: int
    instance_id: int
    Qs: list[np.ndarray]
    I: list[int]


@dataclass
class SolverResult:
    success: bool
    runtime_s: float
    meta: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark separator runtimes over a grid of (k, m) values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--m_values", type=int, nargs="+", required=True, help="Grid of query lengths m.")
    p.add_argument("--k_values", type=int, nargs="+", required=True, help="Grid of center lengths k.")
    p.add_argument("--num_instances", type=int, default=10, help="Instances per (k, m) grid point.")
    p.add_argument("--num_curves", type=int, default=6, help="Number of curves per instance.")
    p.add_argument("--inside_count", type=int, default=3, help="How many curves are labeled inside.")
    p.add_argument(
        "--curve_family",
        type=str,
        choices=["smooth-random", "random", "sine-mixture"],
        default="smooth-random",
        help="Synthetic family used to generate comparable test instances.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible benchmark instances.")
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["softdtw"],
        help="Methods to benchmark. Use 'softdtw' and/or names that include 'qp'.",
    )
    p.add_argument(
        "--external_solver",
        type=str,
        default=None,
        help="Optional module:function path for a second solver to benchmark.",
    )
    p.add_argument(
        "--external_solver_kwargs",
        type=str,
        default="{}",
        help="JSON dictionary of keyword arguments for the external solver.",
    )
    p.add_argument("--gamma", type=float, default=0.1, help="Soft-DTW gamma.")
    p.add_argument("--epochs", type=int, default=1000, help="Soft-DTW epochs.")
    p.add_argument("--retries", type=int, default=5, help="Soft-DTW retries.")
    p.add_argument(
        "--validation",
        action="store_true",
        help="Require hard-DTW validation for the Soft-DTW method.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="benchmark_results",
        help="Directory where detailed and summary CSV files are written.",
    )
    return p.parse_args()


def _smooth_curve(rng: np.random.Generator, m: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, m)
    amps = rng.normal(size=3)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=3)
    freqs = np.array([1.0, 2.0, 3.0])
    curve = sum(a * np.sin(f * math.pi * x + p) for a, f, p in zip(amps, freqs, phases))
    curve += 0.15 * rng.normal(size=m)
    return curve.astype(np.float64)


def _sine_mixture_curve(rng: np.random.Generator, m: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, m)
    warp = np.clip(x ** rng.uniform(0.7, 1.4), 0.0, 1.0)
    curve = (
        rng.uniform(0.6, 1.8) * np.sin(rng.uniform(2.0, 4.0) * math.pi * warp + rng.uniform(-0.5, 0.5))
        + rng.uniform(-0.4, 0.4) * np.cos(rng.uniform(1.0, 3.5) * math.pi * warp + rng.uniform(-0.5, 0.5))
    )
    curve += rng.uniform(-0.2, 0.2) * x
    return curve.astype(np.float64)


def _random_curve(rng: np.random.Generator, m: int) -> np.ndarray:
    return rng.normal(size=m).astype(np.float64)


def _make_curve(rng: np.random.Generator, m: int, family: str) -> np.ndarray:
    if family == "smooth-random":
        return _smooth_curve(rng, m)
    if family == "sine-mixture":
        return _sine_mixture_curve(rng, m)
    return _random_curve(rng, m)


def generate_instances(
    m_values: list[int],
    k_values: list[int],
    num_instances: int,
    num_curves: int,
    inside_count: int,
    curve_family: str,
    seed: int,
) -> list[BenchmarkInstance]:
    if inside_count <= 0 or inside_count >= num_curves:
        raise ValueError("inside_count must satisfy 0 < inside_count < num_curves")

    rng = np.random.default_rng(seed)
    instances: list[BenchmarkInstance] = []
    for m in m_values:
        for k in k_values:
            for instance_id in range(num_instances):
                curves = [_make_curve(rng, m, curve_family) for _ in range(num_curves)]
                perm = rng.permutation(num_curves)
                inside = sorted(int(i) for i in perm[:inside_count])
                instances.append(BenchmarkInstance(m=m, k=k, instance_id=instance_id, Qs=curves, I=inside))
    return instances


def _coerce_external_result(raw: Any) -> tuple[bool, dict[str, Any]]:
    if isinstance(raw, bool):
        return raw, {}
    if isinstance(raw, (tuple, list)) and raw:
        success = bool(raw[0])
        meta = {"raw_return_len": len(raw)}
        return success, meta
    raise TypeError(
        "External solver must return either a bool or a tuple/list whose first element is success."
    )


def load_external_solver(spec: str | None) -> Callable[..., Any] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("external solver spec must be of the form module:function")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_softdtw_solver(instance: BenchmarkInstance, args: argparse.Namespace) -> SolverResult:
    t0 = time.perf_counter()
    success, P_opt, Delta_opt, max_in, min_out, hard_valid = optimize_ball_robust(
        Qs=instance.Qs,
        I=instance.I,
        k=instance.k,
        gamma=args.gamma,
        epochs=args.epochs,
        retries=args.retries,
        require_hard_dtw_validation=args.validation,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return SolverResult(
        success=bool(success),
        runtime_s=elapsed,
        meta={
            "hard_valid": bool(hard_valid),
            "max_in": max_in,
            "min_out": min_out,
            "delta": None if Delta_opt is None else float(Delta_opt.item()),
            "found_center": P_opt is not None,
        },
    )


def run_external_solver(
    instance: BenchmarkInstance,
    solver: Callable[..., Any],
    solver_kwargs: dict[str, Any],
) -> SolverResult:
    t0 = time.perf_counter()
    raw = solver(instance.Qs, instance.I, instance.k, **solver_kwargs)
    elapsed = time.perf_counter() - t0
    success, meta = _coerce_external_result(raw)
    return SolverResult(success=success, runtime_s=elapsed, meta=meta)


def benchmark_methods(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    instances = generate_instances(
        m_values=args.m_values,
        k_values=args.k_values,
        num_instances=args.num_instances,
        num_curves=args.num_curves,
        inside_count=args.inside_count,
        curve_family=args.curve_family,
        seed=args.seed,
    )

    external_solver = load_external_solver(args.external_solver)
    external_solver_kwargs = json.loads(args.external_solver_kwargs)

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        method_lower = method.lower()
        for instance in instances:
            if method_lower == "softdtw":
                result = run_softdtw_solver(instance, args)
            else:
                if external_solver is None:
                    raise ValueError(
                        f"Method '{method}' requested but no --external_solver was provided."
                    )
                result = run_external_solver(instance, external_solver, external_solver_kwargs)

            row = {
                "method": method,
                "m": instance.m,
                "k": instance.k,
                "instance_id": instance.instance_id,
                "num_curves": len(instance.Qs),
                "inside_count": len(instance.I),
                "runtime_s": result.runtime_s,
                "success": int(result.success),
                "seed": args.seed,
                "curve_family": args.curve_family,
            }
            for key, value in result.meta.items():
                row[key] = value
            rows.append(row)

    detailed = pd.DataFrame(rows)
    summary = (
        detailed.groupby(["method", "m", "k"], as_index=False)
        .agg(
            mean_runtime_s=("runtime_s", "mean"),
            median_runtime_s=("runtime_s", "median"),
            std_runtime_s=("runtime_s", "std"),
            min_runtime_s=("runtime_s", "min"),
            max_runtime_s=("runtime_s", "max"),
            success_rate=("success", "mean"),
            num_instances=("runtime_s", "size"),
        )
        .sort_values(["method", "m", "k"])
    )
    return detailed, summary


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    detailed, summary = benchmark_methods(args)

    detailed_path = os.path.join(args.out_dir, "benchmark_detailed.csv")
    summary_path = os.path.join(args.out_dir, "benchmark_summary.csv")
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=" * 72)
    print("Separator runtime benchmark")
    print("=" * 72)
    print(f"Methods       : {args.methods}")
    print(f"m grid        : {args.m_values}")
    print(f"k grid        : {args.k_values}")
    print(f"Instances     : {args.num_instances}")
    print(f"Curve family  : {args.curve_family}")
    print(f"Detailed CSV  : {os.path.abspath(detailed_path)}")
    print(f"Summary CSV   : {os.path.abspath(summary_path)}")
    print("-" * 72)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
