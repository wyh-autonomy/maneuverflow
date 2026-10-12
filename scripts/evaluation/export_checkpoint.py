#!/usr/bin/env python3
"""Export the TF online weights from an EMA Joint Lightning checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="EMA Joint Lightning checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Self-contained inference checkpoint destination.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from navsim.agents.maneuverflow.checkpoint_export import export_joint_checkpoint

    result = export_joint_checkpoint(
        args.input,
        args.output.expanduser().resolve(),
        overwrite=bool(args.force),
    )
    print("ManeuverFlow TF inference checkpoint exported and validated.")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
