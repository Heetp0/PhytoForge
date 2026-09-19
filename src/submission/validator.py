"""Forwarding module for PhytoForge submission validation."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.submission.writer import validate_submission, validate_submission_file


def normalize_format(val: str) -> str:
    if not val or not isinstance(val, str):
        raise argparse.ArgumentTypeError("Format argument cannot be empty.")
    cleaned = val.strip().lower()
    if cleaned in ("inchikey14", "smiles"):
        return cleaned
    raise argparse.ArgumentTypeError(
        f"Invalid format '{val}'. Expected 'inchikey14' or 'smiles'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PhytoForge submission integrity validator")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--expected-ids", type=str, default=None, help="Path to expected test IDs text file")
    parser.add_argument(
        "--format",
        type=normalize_format,
        default=None,
        help="Expected submission format ('inchikey14' or 'smiles'). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"Error: submission file '{args.submission}' not found.", file=sys.stderr)
        sys.exit(1)
    if not sub_path.is_file():
        print(f"Error: submission path '{args.submission}' is not a valid file.", file=sys.stderr)
        sys.exit(1)

    expected_ids = None
    if args.expected_ids:
        exp_path = Path(args.expected_ids)
        if not exp_path.exists():
            print(f"Error: expected-ids file '{args.expected_ids}' not found.", file=sys.stderr)
            sys.exit(1)
        if not exp_path.is_file():
            print(f"Error: expected-ids path '{args.expected_ids}' is not a valid file.", file=sys.stderr)
            sys.exit(1)
        try:
            with open(exp_path, "r", encoding="utf-8", errors="replace") as f:
                expected_ids = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            print(f"Error reading expected-ids file '{args.expected_ids}': {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        is_valid, errors = validate_submission_file(
            sub_path, expected_ids=expected_ids, output_format=args.format
        )
        if not is_valid:
            print(f"Validation failed:\n" + "\n".join(errors), file=sys.stderr)
            sys.exit(1)

        try:
            sub_df = pd.read_csv(sub_path)
        except UnicodeDecodeError:
            sub_df = pd.read_csv(sub_path, encoding="latin-1")

        validate_submission(sub_df, expected_ids=expected_ids, output_format=args.format)
        print(f"Validation successful: {args.submission} conforms to all competition invariants.")
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
