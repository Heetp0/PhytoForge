"""Forwarding module for PhytoForge submission validation."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.submission.writer import validate_submission, validate_submission_file


def main() -> None:
    parser = argparse.ArgumentParser(description="PhytoForge submission integrity validator")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--expected-ids", type=str, default=None, help="Path to expected test IDs text file")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"Error: submission file '{args.submission}' not found.", file=sys.stderr)
        sys.exit(1)

    expected_ids = None
    if args.expected_ids:
        exp_path = Path(args.expected_ids)
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                expected_ids = [line.strip() for line in f if line.strip()]

    try:
        sub_df = pd.read_csv(sub_path)
        validate_submission(sub_df, expected_ids=expected_ids)
        print(f"Validation successful: {args.submission} conforms to all competition invariants.")
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
