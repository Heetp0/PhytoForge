"""Wave C Test Suite: Evaluation Harness & Data Tooling Subsystems.

Verifies:
1. Fast-path InChIKey14 skeletal prefix matching and reciprocal rank calculation.
2. Vectorized / NumPy input handling in calculate_reciprocal_rank.
3. Multi-cutoff aggregate evaluation metrics (MRR@k and Top-K accuracy).
4. Memory-bounded Parquet loading with max_rows slicing in SpectrumLoader.
5. End-to-end benchmark execution and JSON report generation in run_benchmark.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    _matches_ik14,
    calculate_reciprocal_rank,
    evaluate_predictions,
)
from src.evaluation.benchmark import run_benchmark
from src.data.loader import (
    SpectrumLoader,
    SpectrumData,
    QuerySpectrum,
    filter_and_normalize_peaks,
)
from src.pipeline import CASMIOmegaPipeline


def test_metrics_fast_skeletal_prefix_matching():
    """Verify exact, hyphenated, case-insensitive, and mismatched InChIKey comparisons."""
    target = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    cand_skeletal = "BSYNRYMUTXBXSQ"
    cand_lower = "bsynrymutxbxsq-uhfffaoysa-n"
    cand_diff = "IK14DUMMY00001"

    assert _matches_ik14(target, cand_skeletal)
    assert _matches_ik14(target, cand_lower)
    assert _matches_ik14(cand_skeletal, target)
    assert not _matches_ik14(target, cand_diff)
    assert not _matches_ik14("", cand_skeletal)
    assert not _matches_ik14(target, "")
    assert not _matches_ik14(None, cand_skeletal)  # type: ignore


def test_metrics_reciprocal_rank_numpy_and_list():
    """Verify calculate_reciprocal_rank handles lists, tuples, and numpy arrays correctly."""
    target = "REFJWTPEDVJJIY"  # Quercetin InChIKey14
    slots_list = ["OTHER1", "REFJWTPEDVJJIY-UHFFFAOYSA-N", "OTHER2"]
    slots_np = np.array(slots_list)

    # Rank 2 match -> 1/2 = 0.5
    assert calculate_reciprocal_rank(target, slots_list, k=25) == 0.5
    assert calculate_reciprocal_rank(target, slots_np, k=25) == 0.5

    # Rank 1 match -> 1/1 = 1.0
    assert calculate_reciprocal_rank(target, ["REFJWTPEDVJJIY"], k=25) == 1.0

    # Cutoff at k=1 should return 0.0 when match is at rank 2
    assert calculate_reciprocal_rank(target, slots_list, k=1) == 0.0

    # Empty inputs or k <= 0
    assert calculate_reciprocal_rank(target, [], k=25) == 0.0
    assert calculate_reciprocal_rank("", slots_list, k=25) == 0.0
    assert calculate_reciprocal_rank(target, slots_list, k=0) == 0.0


def test_evaluate_predictions_aggregation():
    """Verify evaluate_predictions computes accurate MRR and top-k accuracy across queries."""
    ground_truth = {
        "Q1": "IK14ALPHA00001",
        "Q2": "IK14BETA000002",
        "Q3": "IK14GAMMA00003",
    }
    predictions = {
        "Q1": ["IK14ALPHA00001", "DUMMY1"],          # Rank 1 -> RR = 1.0
        "Q2": ["DUMMY2", "IK14BETA000002"],          # Rank 2 -> RR = 0.5
        "Q3": ["DUMMY3", "DUMMY4", "DUMMY5"],        # Not found -> RR = 0.0
    }

    metrics = evaluate_predictions(ground_truth, predictions, ks=(1, 5, 25))

    # MRR@25 = (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert abs(metrics["mrr@25"] - 0.5) < 1e-5
    # MRR@1 = (1.0 + 0.0 + 0.0) / 3 = 1/3 = 0.333333
    assert abs(metrics["mrr@1"] - (1.0 / 3.0)) < 1e-5
    # Top-1 accuracy = 1 / 3
    assert abs(metrics["top1_accuracy"] - (1.0 / 3.0)) < 1e-5
    # Top-5 accuracy = 2 / 3
    assert abs(metrics["top5_accuracy"] - (2.0 / 3.0)) < 1e-5


def test_spectrum_loader_parquet_streaming(tmp_path: Path):
    """Verify SpectrumLoader.load_parquet handles max_rows slicing correctly."""
    records = []
    for i in range(10):
        records.append(
            SpectrumData(
                molecule_id=f"SPEC_{i:03d}",
                precursor_mz=200.0 + i,
                adduct="[M+H]+",
                polarity="positive",
                collision_energy=30.0,
                mz_array=np.array([100.0 + i, 150.0 + i]),
                intensity_array=np.array([50.0, 100.0]),
                inchikey14=f"IK14TEST{i:06d}",
            )
        )

    parquet_file = tmp_path / "test_spectra.parquet"
    SpectrumLoader.save_parquet(records, parquet_file)

    # Load bounded 3 rows
    loaded = SpectrumLoader.load_parquet(parquet_file, max_rows=3)
    assert len(loaded) == 3
    assert loaded[0].molecule_id == "SPEC_000"
    assert loaded[2].molecule_id == "SPEC_002"

    # Convert to query spectra
    q_specs = SpectrumLoader.to_query_spectra(loaded)
    assert len(q_specs) == 3
    assert isinstance(q_specs[0], QuerySpectrum)


def test_benchmark_execution_and_reporting(tmp_path: Path):
    """Verify run_benchmark executes pipeline, calculates metrics, and exports report."""
    test_df = pd.DataFrame([
        {
            "id": "BENCH_001",
            "precursor_mz": 303.0496,
            "adduct": "[M+H]+",
            "polarity": "positive",
            "peaks": "153.019:100.0;303.049:50.0",
            "inchikey14": "IK14DBFLAV0001",
        }
    ])

    report_file = tmp_path / "benchmark_report.json"
    report = run_benchmark(
        df=test_df,
        output_path=report_file,
    )

    assert "total_spectra" in report
    assert report["total_spectra"] == 1
    assert "mrr@25" in report
    assert "avg_latency_ms" in report
    assert "governor_compliant" in report
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report["total_spectra"] == 1
