"""Forwarding module for PhytoForge pipeline execution."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.pipeline import CASMIOmegaPipeline, PhytoForgePipeline, run_phytoforge_pipeline


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
    parser = argparse.ArgumentParser(description="PhytoForge inference pipeline runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input test parquet or csv")
    parser.add_argument("--output", type=str, required=True, help="Path to output submission CSV")
    parser.add_argument(
        "--format",
        type=normalize_format,
        default="inchikey14",
        help="Submission output format: 'inchikey14' (id,candidates) or 'smiles' (molecule_id,smiles)",
    )
    parser.add_argument("--governor", type=float, default=21.3, help="Max budget per spectrum in seconds")
    parser.add_argument("--vram-cap", type=float, default=16.0, help="VRAM cap in GB")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    if not input_path.is_file():
        print(f"Error: input path '{args.input}' is not a valid file.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.exists() and output_path.is_dir():
        print(f"Error: output path '{args.output}' is a directory, not a file.", file=sys.stderr)
        sys.exit(1)

    try:
        if input_path.suffix == ".parquet":
            test_df = pd.read_parquet(input_path)
        else:
            try:
                test_df = pd.read_csv(input_path)
            except UnicodeDecodeError:
                test_df = pd.read_csv(input_path, encoding="latin-1")
    except Exception as exc:
        print(f"Error reading input dataset '{args.input}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        budget = args.governor * len(test_df) if not test_df.empty else 32400.0
        run_phytoforge_pipeline(
            test_df,
            output_path=args.output,
            output_format=args.format,
            governor_budget_seconds=budget,
        )
        print(f"Pipeline completed successfully. Output written to {args.output}")
    except Exception as exc:
        print(f"Pipeline execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
