#!/usr/bin/env python3
"""Run the preregistered SRRD Phase 1 falsification experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from srrd_phase1.experiment import ExperimentConfig, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--seeds", type=int, default=400, help="Seeds per scenario")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=4000, help="Bootstrap draws"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig(
        n_seeds=args.seeds,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result = run_experiment(config, ROOT / args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gates"]["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

