"""Forwarding module for PhytoForge pipeline execution."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.pipeline import CASMIOmegaPipeline, PhytoForgePipeline, run_phytoforge_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="PhytoForge inference pipeline runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input test parquet or mgf")
    parser.add_argument("--output", type=str, required=True, help="Path to output submission CSV")
    parser.add_argument("--governor", type=float, default=21.3, help="Max budget per spectrum in seconds")
    parser.add_argument("--vram-cap", type=float, default=16.0, help="VRAM cap in GB")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix == ".parquet":
        test_df = pd.read_parquet(input_path)
    else:
        test_df = pd.read_csv(input_path)

    run_phytoforge_pipeline(test_df, output_path=args.output)
    print(f"Pipeline completed successfully. Output written to {args.output}")


if __name__ == "__main__":
    main()
