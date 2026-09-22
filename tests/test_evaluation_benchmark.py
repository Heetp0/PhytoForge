"""
tests/test_evaluation_benchmark.py

Comprehensive test suite for:
- run_benchmark function in src/evaluation/benchmark.py
- Latency measurement and dynamic runtime governor compliance
- MRR@25 and Top-K accuracies (1, 5, 10, 25)
- Track recalls breakdown across all 5 tracks (track1_dreams, track2_db, track3_denovo, track_network, track_knapsack, any)
- Empty DataFrame edge case handling
- Temporary submission CSV handling (JSON output never clobbered)
- External pipeline lifecycle (reused without closing) vs internal context manager
- Ground truth extraction from inchikey14, full inchikey, and smiles
- CLI entrypoint (python -m src.evaluation.benchmark)
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import pandas as pd
import pytest

from src.evaluation.benchmark import run_benchmark
from src.pipeline import CASMIOmegaPipeline


@pytest.fixture
def mock_evaluation_dataframe() -> pd.DataFrame:
    """Creates a 2-spectrum synthetic evaluation dataframe with ground-truth targets."""
    return pd.DataFrame({
        "id": ["SPEC_0001", "SPEC_0002"],
        "precursor_mz": [286.047, 300.0],
        "polarity": ["positive", "positive"],
        "peaks": ["100.0:10.0;200.0:20.0", "150.0:5.0;250.0:15.0"],
        # SPEC_0001 matches default pipeline DB compound IK14DBFLAV0001 (formula C15H10O5)
        # SPEC_0002 matches default fallback scaffold IK14FALL000000
        "inchikey14": ["IK14DBFLAV0001", "IK14FALL000000"],
    })


class TestRunBenchmark:
    """Test run_benchmark function execution, metrics, and lifecycle."""

    def test_empty_dataframe_returns_zero_metrics(self) -> None:
        report = run_benchmark(pd.DataFrame())

        assert report["total_spectra"] == 0
        assert report["total_time_seconds"] == 0.0
        assert report["avg_latency_ms"] == 0.0
        assert report["governor_compliant"] is True
        assert report["mrr@25"] == 0.0
        assert report["top1_accuracy"] == 0.0
        assert report["top25_accuracy"] == 0.0

        for track in ("track1_dreams", "track2_db", "track3_denovo", "track_network", "track_knapsack", "any"):
            assert report["track_recalls"][track] == 0.0

    def test_empty_dataframe_writes_json_report(self, tmp_path: Path) -> None:
        out_json = tmp_path / "empty_report.json"
        report = run_benchmark(pd.DataFrame(), output_path=out_json)

        assert out_json.exists()
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["total_spectra"] == 0
        assert loaded["mrr@25"] == 0.0
        assert loaded == report

    def test_mock_dataframe_metrics_and_governor(self, mock_evaluation_dataframe: pd.DataFrame, tmp_path: Path) -> None:
        out_json = tmp_path / "benchmark_report.json"
        report = run_benchmark(mock_evaluation_dataframe, output_path=out_json)

        assert report["total_spectra"] == 2
        assert report["total_time_seconds"] > 0.0
        assert report["avg_latency_ms"] > 0.0
        assert report["governor_compliant"] is True

        # Pipeline matches IK14DBFLAV0001 and IK14FALL000000, so MRR and Top-25 accuracy must be positive
        assert report["mrr@25"] > 0.0
        assert report["top25_accuracy"] > 0.0

        # Track recalls verification
        assert "track_recalls" in report
        recalls = report["track_recalls"]
        assert "track2_db" in recalls
        assert recalls["track2_db"] > 0.0  # SPEC_0001 is retrieved from Track 2 DB
        assert recalls["any"] > 0.0

        # Output JSON file verification
        assert out_json.exists()
        raw_text = out_json.read_text(encoding="utf-8")
        assert not raw_text.startswith("id,candidates")  # Must NOT be CSV
        loaded = json.loads(raw_text)
        assert loaded["total_spectra"] == 2
        assert loaded["governor_compliant"] is True

    def test_pre_instantiated_pipeline_lifecycle(self, mock_evaluation_dataframe: pd.DataFrame) -> None:
        pipeline = CASMIOmegaPipeline()
        try:
            report = run_benchmark(mock_evaluation_dataframe, pipeline=pipeline)
            assert report["total_spectra"] == 2
            # Verify pipeline is still open and contains accumulated candidates
            assert hasattr(pipeline, "all_aggregated_cands")
            assert "SPEC_0001" in pipeline.all_aggregated_cands
            assert "SPEC_0002" in pipeline.all_aggregated_cands
            assert len(pipeline.all_aggregated_cands["SPEC_0001"]) > 0
        finally:
            pipeline.close()

    def test_ground_truth_extraction_from_smiles(self) -> None:
        smi = "c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1"
        df = pd.DataFrame({
            "id": ["SPEC_0001"],
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0;200.0:20.0"],
            # Smiles for candidate
            "smiles": [smi],
        })
        report = run_benchmark(df, output_format="smiles")
        assert report["total_spectra"] == 1
        assert report["mrr@25"] > 0.0

    def test_ground_truth_extraction_from_full_inchikey(self) -> None:
        df = pd.DataFrame({
            "id": ["SPEC_0001"],
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0;200.0:20.0"],
            # Full hyphenated InChIKey
            "inchikey": ["IK14DBFLAV0001-UHFFFAOYSA-N"],
        })
        report = run_benchmark(df)
        assert report["total_spectra"] == 1
        assert report["mrr@25"] > 0.0


class TestBenchmarkCLI:
    """Test CLI invocation of python -m src.evaluation.benchmark."""

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluation.benchmark", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output" in result.stdout

    def test_cli_execution_csv(self, mock_evaluation_dataframe: pd.DataFrame, tmp_path: Path) -> None:
        test_csv = tmp_path / "test_queries.csv"
        mock_evaluation_dataframe.to_csv(test_csv, index=False)
        out_json = tmp_path / "cli_report.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.benchmark",
                "--input",
                str(test_csv),
                "--output",
                str(out_json),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
        assert out_json.exists()
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["total_spectra"] == 2
        assert "total_spectra" in result.stdout

    def test_cli_execution_parquet(self, mock_evaluation_dataframe: pd.DataFrame, tmp_path: Path) -> None:
        test_pq = tmp_path / "test_queries.parquet"
        mock_evaluation_dataframe.to_parquet(test_pq, index=False)
        out_json = tmp_path / "cli_pq_report.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.benchmark",
                "--input",
                str(test_pq),
                "--output",
                str(out_json),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
        assert out_json.exists()

    def test_cli_missing_input_exits_1(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.csv"
        out_json = tmp_path / "report.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.benchmark",
                "--input",
                str(non_existent),
                "--output",
                str(out_json),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr
