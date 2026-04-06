from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .compare import batch_compare, compare_trace, write_trace_artifacts
from .inference import BayesianConfig, fit_bayesian
from .io import load_manifest, load_trace
from .legacy import run_legacy


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian Langmuir probe analysis with the Orsini EEDF model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("fit", "compare"):
        subparser = subparsers.add_parser(name)
        _add_common_arguments(subparser)
        subparser.set_defaults(func=_run_fit if name == "fit" else _run_compare)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("manifest", type=Path)
    batch_parser.add_argument("--trace-id", action="append", dest="trace_ids")
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    _add_config_arguments(batch_parser)
    batch_parser.set_defaults(func=_run_batch)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    _add_config_arguments(parser)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nlive", type=int, default=125)
    parser.add_argument("--dlogz", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--posterior-draws", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=7)


def _config_from_args(args: argparse.Namespace) -> BayesianConfig:
    return BayesianConfig(
        nlive=args.nlive,
        dlogz=args.dlogz,
        max_points=args.max_points,
        posterior_draws=args.posterior_draws,
        random_seed=args.random_seed,
    )


def _load_trace_from_manifest(manifest_path: Path, trace_id: str):
    manifest = load_manifest(manifest_path)
    manifest["manifest_path"] = manifest.attrs["manifest_path"]
    selected = manifest.loc[manifest["trace_id"] == trace_id]
    if selected.empty:
        raise SystemExit(f"Trace {trace_id!r} was not found in manifest {manifest_path}.")
    return load_trace(selected.iloc[0])


def _run_fit(args: argparse.Namespace) -> None:
    trace = _load_trace_from_manifest(args.manifest, args.trace_id)
    bayes = fit_bayesian(trace, config=_config_from_args(args))
    comparison = compare_trace(trace, bayes, None)
    write_trace_artifacts(args.output_dir, trace, bayes, None, comparison)
    print(bayes.summary.to_string())


def _run_compare(args: argparse.Namespace) -> None:
    trace = _load_trace_from_manifest(args.manifest, args.trace_id)
    bayes = fit_bayesian(trace, config=_config_from_args(args))
    legacy = run_legacy(trace)
    comparison = compare_trace(trace, bayes, legacy)
    write_trace_artifacts(args.output_dir, trace, bayes, legacy, comparison)
    print(comparison.comparison_table.to_string())


def _run_batch(args: argparse.Namespace) -> None:
    summary = batch_compare(
        manifest_path=args.manifest,
        trace_ids=args.trace_ids,
        config=_config_from_args(args),
        output_dir=args.output_dir,
    )
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
