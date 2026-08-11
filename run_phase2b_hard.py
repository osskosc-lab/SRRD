#!/usr/bin/env python3
"""Run SRRD Phase 2B-hard smoke, development, or confirmatory experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from srrd_phase2b_hard import Phase2BHardConfig, run_phase2b_hard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development", "confirmatory"), default="smoke")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def config_for(mode: str) -> Phase2BHardConfig:
    if mode == "smoke":
        return Phase2BHardConfig(
            base_seeds=1,
            confirmation_seeds=1,
            confirmation_scenarios=(),
            n_train=128,
            n_test=64,
            tuning_trials=1,
            max_epochs=3,
            patience=2,
            state_permutations=31,
            bootstrap_replicates=200,
        )
    if mode == "development":
        return Phase2BHardConfig(
            base_seeds=4,
            confirmation_seeds=4,
            confirmation_scenarios=(),
            n_train=256,
            n_test=128,
            tuning_trials=2,
            max_epochs=10,
            patience=4,
            state_permutations=99,
            bootstrap_replicates=500,
        )
    return Phase2BHardConfig()


def main() -> int:
    args = parse_args()
    config = config_for(args.mode)
    output = args.output or f"results/phase2b_hard_{args.mode}"
    result = run_phase2b_hard(
        config,
        ROOT / output,
        preregistration_path=ROOT / "preregistration" / "phase2b_hard.yaml",
        freeze_path=ROOT / "preregistration" / "phase2b_hard.freeze.json",
        require_frozen=args.mode == "confirmatory",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 3 if result["classification"] == "F_implementation_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
