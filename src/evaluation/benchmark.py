"""
End-to-End Evaluation & Latency Benchmark Harness for PhytoForge CASMI 2026.

Provides:
- run_benchmark: Executes CASMIOmegaPipeline on query spectra, computes MRR@25, Top-K accuracies
  (1, 5, 10, 25), multi-track retrieval recalls (track1_dreams, track2_db, track3_denovo,
  track_network, track_knapsack, any), average per-spectrum latency in milliseconds, and
  governor compliance (21.3s/spectrum limit). Writes JSON report without clobbering.
- CLI entrypoint: python -m src.evaluation.benchmark --input <path> --output <report.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.evaluation.metrics import _matches_ik14, evaluate_predictions
from src.pipeline import CASMIOmegaPipeline
from src.submission.writer import get_inchikey14

TRACK_NAMES = [
    "track1_dreams",
    "track2_db",
    "track3_denovo",
    "track_network",
    "track_knapsack",
]


def _matches_cand_target(target: str, cand_dict: Dict[str, Any]) -> bool:
    """Check if candidate dictionary matches ground truth target by InChIKey14 or SMILES."""
    ik14 = cand_dict.get("inchikey14")
    if ik14 and _matches_ik14(target, str(ik14).strip()):
        return True
    smi = cand_dict.get("smiles")
    if smi:
        derived = get_inchikey14(str(smi).strip())
        if derived and _matches_ik14(target, derived):
            return True
    return False


def run_benchmark(
    df: pd.DataFrame,
    pipeline: Optional[CASMIOmegaPipeline] = None,
    output_path: Optional[Union[str, Path]] = None,
    output_format: str = "inchikey14",
    **pipeline_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run end-to-end evaluation & latency benchmark on query spectra.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataset containing query spectra and ground truth columns
        ('inchikey14', 'inchikey', or 'smiles').
    pipeline : Optional[CASMIOmegaPipeline], default=None
        Optional pre-instantiated pipeline. If None, instantiates CASMIOmegaPipeline
        as a context manager with pipeline_kwargs.
    output_path : Optional[Union[str, Path]], default=None
        Optional path to write formatted JSON benchmark report.
    output_format : str, default='inchikey14'
        Submission format ('inchikey14' or 'smiles').
    **pipeline_kwargs : Any
        Keyword arguments passed to CASMIOmegaPipeline constructor if pipeline is None.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing benchmark evaluation metrics:
        - total_spectra
        - total_time_seconds
        - avg_latency_ms
        - governor_compliant
        - mrr@25
        - top1_accuracy, top5_accuracy, top10_accuracy, top25_accuracy
        - track_recalls (per-track breakdown and 'any')
    """
    # 1. Handle empty DataFrame edge case
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        report: Dict[str, Any] = {
            "total_spectra": 0,
            "total_time_seconds": 0.0,
            "avg_latency_ms": 0.0,
            "governor_compliant": True,
            "mrr@25": 0.0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "top25_accuracy": 0.0,
            "track_recalls": {
                "track1_dreams": 0.0,
                "track2_db": 0.0,
                "track3_denovo": 0.0,
                "track_network": 0.0,
                "track_knapsack": 0.0,
                "any": 0.0,
            },
        }
        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    n_spectra = len(df)

    # 2. Extract ground truth targets per query
    ground_truth: Dict[str, str] = {}
    id_col = "id" if "id" in df.columns else ("molecule_id" if "molecule_id" in df.columns else None)

    for idx, (_, row) in enumerate(df.iterrows()):
        qid = str(row[id_col]) if id_col is not None and pd.notna(row.get(id_col)) else f"SPEC_{idx:04d}"

        target = ""
        if "inchikey14" in df.columns and pd.notna(row.get("inchikey14")):
            target = str(row["inchikey14"]).strip()
        elif "inchikey" in df.columns and pd.notna(row.get("inchikey")):
            raw_ik = str(row["inchikey"]).strip()
            target = raw_ik.split("-")[0] if "-" in raw_ik else raw_ik
        elif "smiles" in df.columns and pd.notna(row.get("smiles")):
            derived_ik = get_inchikey14(str(row["smiles"]).strip())
            if derived_ik:
                target = derived_ik

        if target:
            ground_truth[qid] = target

    # 3. Execute pipeline using a temporary submission CSV so output_path JSON is never clobbered
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_csv = Path(tmp_dir) / "benchmark_submission.csv"

        t0 = time.perf_counter()
        if pipeline is not None:
            active_pipeline = pipeline
            sub_df = active_pipeline.run(
                test_df=df,
                output_path=temp_csv,
                output_format=output_format,
            )
        else:
            with CASMIOmegaPipeline(**pipeline_kwargs) as pipe:
                active_pipeline = pipe
                sub_df = pipe.run(
                    test_df=df,
                    output_path=temp_csv,
                    output_format=output_format,
                )
        total_time = time.perf_counter() - t0

    # 4. Latency & Governor calculations
    avg_latency_ms = (total_time / n_spectra) * 1000.0 if n_spectra > 0 else 0.0
    governor_compliant = (total_time / n_spectra) <= 21.3 if n_spectra > 0 else True

    # 5. Parse predictions from submission dataframe
    predictions: Dict[str, List[str]] = {}
    pred_id_col = "id" if "id" in sub_df.columns else ("molecule_id" if "molecule_id" in sub_df.columns else None)
    cand_col = "candidates" if "candidates" in sub_df.columns else ("smiles" if "smiles" in sub_df.columns else None)

    if pred_id_col is not None and cand_col is not None:
        for _, row in sub_df.iterrows():
            qid = str(row[pred_id_col])
            raw_cands = str(row[cand_col])
            raw_list = [c.strip() for c in raw_cands.split(";") if c.strip()]
            if cand_col == "smiles" or output_format == "smiles":
                cands = [get_inchikey14(c) or c for c in raw_list]
            else:
                cands = raw_list
            predictions[qid] = cands

    # 6. Compute MRR@25 and Top-K Accuracies
    metrics = evaluate_predictions(ground_truth, predictions, ks=(1, 5, 10, 25))

    # 7. Compute Track Recalls using accumulated candidates
    all_cands_map = getattr(active_pipeline, "all_aggregated_cands", {})
    track_hits: Dict[str, int] = {t: 0 for t in TRACK_NAMES}
    any_hits = 0
    gt_evaluated = 0

    for qid, target in ground_truth.items():
        gt_evaluated += 1
        cands_for_q = all_cands_map.get(qid, [])

        q_track_hit = {t: False for t in TRACK_NAMES}
        q_any_hit = False

        for cand in cands_for_q:
            if _matches_cand_target(target, cand):
                q_any_hit = True
                src = cand.get("source")
                if src in q_track_hit:
                    q_track_hit[src] = True

        if q_any_hit:
            any_hits += 1
        for t in TRACK_NAMES:
            if q_track_hit[t]:
                track_hits[t] += 1

    total_gt = gt_evaluated if gt_evaluated > 0 else len(ground_truth)
    track_recalls: Dict[str, float] = {}
    for t in TRACK_NAMES:
        track_recalls[t] = float(track_hits[t] / total_gt) if total_gt > 0 else 0.0
    track_recalls["any"] = float(any_hits / total_gt) if total_gt > 0 else 0.0

    # 8. Build report dictionary
    report = {
        "total_spectra": n_spectra,
        "total_time_seconds": round(float(total_time), 4),
        "avg_latency_ms": round(float(avg_latency_ms), 2),
        "governor_compliant": bool(governor_compliant),
        "mrr@25": float(metrics.get("mrr@25", 0.0)),
        "top1_accuracy": float(metrics.get("top1_accuracy", metrics.get("top_1_accuracy", 0.0))),
        "top5_accuracy": float(metrics.get("top5_accuracy", metrics.get("top_5_accuracy", 0.0))),
        "top10_accuracy": float(metrics.get("top10_accuracy", metrics.get("top_10_accuracy", 0.0))),
        "top25_accuracy": float(metrics.get("top25_accuracy", metrics.get("top_25_accuracy", 0.0))),
        "track_recalls": track_recalls,
    }

    # 9. Output JSON report if requested
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    """CLI entrypoint for end-to-end evaluation and latency benchmark."""
    parser = argparse.ArgumentParser(
        description="PhytoForge End-to-End Evaluation & Latency Benchmark CLI"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input test parquet or csv with spectra and ground-truth",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output JSON benchmark report",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="inchikey14",
        choices=["inchikey14", "smiles"],
        help="Output submission format",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional path to candidate SQLite database",
    )
    parser.add_argument(
        "--spectral-index-path",
        type=str,
        default=None,
        help="Optional path to DreaMS spectral index",
    )
    parser.add_argument(
        "--fragment-library-path",
        type=str,
        default=None,
        help="Optional path to fragment library .pkl",
    )
    parser.add_argument(
        "--meta-ranker-path",
        type=str,
        default=None,
        help="Optional path to trained LightGBM meta-ranker .txt",
    )
    parser.add_argument(
        "--governor-budget",
        type=float,
        default=32400.0,
        help="Total governor budget in seconds",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    if not input_path.is_file():
        print(f"Error: input path '{args.input}' is not a valid file.", file=sys.stderr)
        sys.exit(1)

    try:
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            try:
                df = pd.read_csv(input_path)
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding="latin-1")
    except Exception as exc:
        print(f"Error loading input dataset '{args.input}': {exc}", file=sys.stderr)
        sys.exit(1)

    # Optional fragment library loading
    frag_lib = None
    if args.fragment_library_path:
        try:
            import pickle
            from src.reranking.fragment_library import NeutralLossLibrary
            with open(args.fragment_library_path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, NeutralLossLibrary):
                    frag_lib = loaded.library
                elif isinstance(loaded, dict):
                    frag_lib = loaded
        except Exception as exc:
            print(f"Warning: could not load fragment library: {exc}", file=sys.stderr)

    pipeline_kwargs: Dict[str, Any] = {
        "db_path": args.db_path,
        "spectral_index_path": args.spectral_index_path,
        "fragment_library": frag_lib,
        "governor_budget_seconds": args.governor_budget,
    }

    try:
        report = run_benchmark(
            df=df,
            output_path=args.output,
            output_format=args.format,
            **pipeline_kwargs,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0)
    except Exception as exc:
        print(f"Benchmark execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
