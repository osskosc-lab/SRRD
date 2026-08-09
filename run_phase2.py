#!/usr/bin/env python3
"""Run SRRD Phase 2 development or confirmatory experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from srrd_phase2.experiment import Phase2Config, run_phase2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase2_confirmatory")
    parser.add_argument(
        "--mode", choices=("development", "confirmatory"), default="confirmatory"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "development":
        config = Phase2Config(
            base_seeds=50,
            confirmation_seeds=50,
            confirmation_scenarios=(),
            bootstrap_replicates=args.bootstrap_replicates or 1000,
            rotation_seeds=20,
        )
    else:
        config = Phase2Config(
            bootstrap_replicates=args.bootstrap_replicates or 4000
        )
    result = run_phase2(
        config,
        ROOT / args.output,
        preregistration_path=ROOT / "preregistration" / "phase2.yaml",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["classification"] != "F_implementation_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
