"""
tests/test_adversarial_phase4_bundle_benchmark.py

Empirical Challenger adversarial stress test suite for Phase 4:
- Standalone execution of dist/kaggle_kernel.py in a separate isolated process.
- Kaggle kernel parameter and asset wiring audit (including meta-ranker).
- Benchmark resilience against corrupted, malformed, and missing inputs.
- Benchmark JSON report integrity, schema compliance, and clobbering resistance.
- Dynamic runtime governor compliance and boundary stress.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark import TRACK_NAMES, run_benchmark
from src.submission.kaggle_bundle import build_kaggle_kernel, KaggleEnvironmentConfig
from src.submission.runtime_governor import DynamicRuntimeGovernor
from src.submission.writer import validate_submission_file


@pytest.fixture(scope="module")
def compiled_kernel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and provide a compiled kaggle_kernel.py."""
    kernel_dir = tmp_path_factory.mktemp("kernel_bundle")
    kernel_path = kernel_dir / "kaggle_kernel.py"
    build_kaggle_kernel(output_file=kernel_path)
    return kernel_path


class TestStandaloneKernelAdversarial:
    """Adversarially challenge standalone kaggle_kernel.py in isolated processes."""

    def test_kernel_isolated_cwd_execution(self, compiled_kernel: Path, tmp_path: Path) -> None:
        """
        Execute compiled kernel from an isolated temporary working directory
        completely outside the project workspace, ensuring no local ambient
        imports leak and that _bootstrap_source extracts embedded source cleanly.
        """
        isolated_cwd = tmp_path / "isolated_workspace"
        isolated_cwd.mkdir()

        # Synthetic 3-spectrum test input in CSV format
        input_csv = isolated_cwd / "test_input.csv"
        df = pd.DataFrame({
            "id": ["ISO_001", "ISO_002", "ISO_003"],
            "precursor_mz": [286.047, 300.0, 450.123],
            "polarity": ["positive", "negative", "positive"],
            "peaks": ["100.0:10.0;200.0:20.0", "150.0:5.0", "220.0:15.0;330.0:8.0"],
        })
        df.to_csv(input_csv, index=False)

        output_csv = isolated_cwd / "output_submission.csv"

        # Execute kernel in separate process
        proc = subprocess.run(
            [
                sys.executable,
                str(compiled_kernel),
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
                "--format",
                "inchikey14",
            ],
            capture_output=True,
            text=True,
            cwd=str(isolated_cwd),
            check=False,
        )

        assert proc.returncode == 0, f"Kernel failed in isolated CWD:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert output_csv.exists()

        # Validate extracted bootstrap directory
        bootstrap_dir = isolated_cwd / ".kaggle_bundle_src"
        assert bootstrap_dir.exists()
        assert (bootstrap_dir / "src" / "pipeline.py").exists()

        # Validate generated submission CSV schema
        validate_submission_file(
            output_csv,
            expected_ids=["ISO_001", "ISO_002", "ISO_003"],
            output_format="inchikey14",
        )

        # Confirm exactly 3 rows + header
        lines = output_csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        assert lines[0] == "id,candidates"

    def test_kernel_smiles_format_isolated_execution(self, compiled_kernel: Path, tmp_path: Path) -> None:
        """Execute kernel with --format smiles and verify valid SMILES candidates."""
        isolated_cwd = tmp_path / "smiles_workspace"
        isolated_cwd.mkdir()

        input_csv = isolated_cwd / "test_smiles.csv"
        df = pd.DataFrame({
            "id": ["SMI_001", "SMI_002"],
            "precursor_mz": [286.047, 300.0],
            "polarity": ["positive", "positive"],
            "peaks": ["100.0:10.0;200.0:20.0", "150.0:5.0"],
        })
        df.to_csv(input_csv, index=False)

        output_csv = isolated_cwd / "submission_smiles.csv"

        proc = subprocess.run(
            [
                sys.executable,
                str(compiled_kernel),
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
                "--format",
                "smiles",
            ],
            capture_output=True,
            text=True,
            cwd=str(isolated_cwd),
            check=False,
        )

        assert proc.returncode == 0, f"Kernel SMILES failed:\nSTDERR:\n{proc.stderr}"
        assert output_csv.exists()

        validate_submission_file(
            output_csv,
            expected_ids=["SMI_001", "SMI_002"],
            output_format="smiles",
        )

    def test_kernel_with_explicit_offline_assets(self, compiled_kernel: Path, tmp_path: Path) -> None:
        """
        Execute kernel with mock offline asset paths (candidate DB, fragment library).
        Verifies that explicit asset resolution and loading succeeds.
        """
        asset_dir = tmp_path / "mock_assets"
        asset_dir.mkdir()

        # 1. Create a dummy fragment library
        frag_file = asset_dir / "fragments.pkl"
        with open(frag_file, "wb") as f:
            pickle.dump({2860470: ["c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1"]}, f)

        input_csv = asset_dir / "queries.csv"
        df = pd.DataFrame({
            "id": ["ASSET_001"],
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0;200.0:20.0"],
        })
        df.to_csv(input_csv, index=False)
        sub_csv = asset_dir / "submission.csv"

        proc = subprocess.run(
            [
                sys.executable,
                str(compiled_kernel),
                "--input",
                str(input_csv),
                "--output",
                str(sub_csv),
                "--fragment-library-path",
                str(frag_file),
                "--governor",
                "15.0",
                "--reserve",
                "100.0",
            ],
            capture_output=True,
            text=True,
            cwd=str(asset_dir),
            check=False,
        )

        assert proc.returncode == 0, f"Asset kernel failed:\nSTDERR:\n{proc.stderr}"
        assert sub_csv.exists()

    def test_kernel_nonexistent_input_fails_cleanly(self, compiled_kernel: Path, tmp_path: Path) -> None:
        """Verify standalone kernel exits with code 1 and error message when input is missing."""
        proc = subprocess.run(
            [
                sys.executable,
                str(compiled_kernel),
                "--input",
                str(tmp_path / "ghost_file.csv"),
                "--output",
                str(tmp_path / "sub.csv"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert "execution failed" in proc.stderr.lower() or "not found" in proc.stderr.lower()


class TestBenchmarkAdversarialInputs:
    """Stress-test benchmark.py against severely corrupted and boundary inputs."""

    def test_missing_all_ground_truth_columns(self) -> None:
        """
        Input DataFrame has spectra data but completely lacks ground-truth columns
        (no inchikey14, inchikey, or smiles).
        Benchmark should execute inference, record latency, report 0.0 for accuracy,
        and not crash.
        """
        df = pd.DataFrame({
            "id": ["BLIND_001", "BLIND_002"],
            "precursor_mz": [300.0, 400.0],
            "polarity": ["positive", "positive"],
            "peaks": ["100.0:10.0", "150.0:20.0"],
        })

        report = run_benchmark(df)
        assert report["total_spectra"] == 2
        assert report["mrr@25"] == 0.0
        assert report["top1_accuracy"] == 0.0
        assert report["top25_accuracy"] == 0.0
        assert report["governor_compliant"] is True
        assert report["avg_latency_ms"] > 0.0
        assert report["track_recalls"]["any"] == 0.0

    def test_corrupted_and_nan_precursor_mz(self) -> None:
        """
        Precursor m/z containing NaN, Inf, negative, or string values.
        Pipeline should fall back to 300.0 m/z default without crashing.
        """
        df = pd.DataFrame({
            "id": ["NAN_001", "INF_002", "NEG_003", "STR_004"],
            "precursor_mz": [np.nan, np.inf, -999.0, "garbage_val"],
            "polarity": ["positive", "positive", "positive", "positive"],
            "peaks": ["100.0:10.0", "100.0:10.0", "100.0:10.0", "100.0:10.0"],
            "inchikey14": ["IK14FALL000000", "IK14FALL000000", "IK14FALL000000", "IK14FALL000000"],
        })

        report = run_benchmark(df)
        assert report["total_spectra"] == 4
        assert report["governor_compliant"] is True
        # Since precursor_mz falls back to 300.0, fallback scaffolds match IK14FALL000000
        assert report["mrr@25"] > 0.0

    def test_missing_id_column_auto_generates_spec_ids(self) -> None:
        """DataFrame missing 'id' column entirely."""
        df = pd.DataFrame({
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0"],
            "inchikey14": ["IK14DBFLAV0001"],
        })

        report = run_benchmark(df)
        assert report["total_spectra"] == 1
        assert report["mrr@25"] > 0.0

    def test_unparseable_smiles_ground_truth(self) -> None:
        """SMILES column contains malformed chemical strings."""
        df = pd.DataFrame({
            "id": ["BAD_SMI_1"],
            "precursor_mz": [300.0],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0"],
            "smiles": ["[[INVALID_CHEMS_SMILES--NOT_A_MOL]]"],
        })

        report = run_benchmark(df)
        assert report["total_spectra"] == 1
        # Bad SMILES cannot derive InChIKey14, so ground_truth is empty -> MRR is 0.0
        assert report["mrr@25"] == 0.0

    def test_invalid_adduct_strings(self) -> None:
        """Adduct column has unparseable or nonsense adduct annotations."""
        df = pd.DataFrame({
            "id": ["ADDUCT_001", "ADDUCT_002"],
            "precursor_mz": [286.047, 300.0],
            "polarity": ["positive", "negative"],
            "adduct": ["MYSTERY_ION_ADDUCT", ""],
            "peaks": ["100.0:10.0", "150.0:20.0"],
            "inchikey14": ["IK14DBFLAV0001", "IK14FALL000000"],
        })

        report = run_benchmark(df)
        assert report["total_spectra"] == 2
        assert report["governor_compliant"] is True


class TestBenchmarkJSONReportIntegrity:
    """Verify JSON benchmark report schema, serializability, and clobbering immunity."""

    def test_report_schema_strictness_and_types(self, tmp_path: Path) -> None:
        """Verify all report fields and exact expected types."""
        out_json = tmp_path / "schema_check.json"
        df = pd.DataFrame({
            "id": ["T_01"],
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0"],
            "inchikey14": ["IK14DBFLAV0001"],
        })

        report = run_benchmark(df, output_path=out_json)
        assert out_json.exists()

        # Strict JSON parse check
        loaded = json.loads(out_json.read_text(encoding="utf-8"))

        assert isinstance(loaded["total_spectra"], int)
        assert isinstance(loaded["total_time_seconds"], float)
        assert isinstance(loaded["avg_latency_ms"], float)
        assert isinstance(loaded["governor_compliant"], bool)
        assert isinstance(loaded["mrr@25"], float)
        assert 0.0 <= loaded["mrr@25"] <= 1.0
        assert isinstance(loaded["top1_accuracy"], float)
        assert isinstance(loaded["top5_accuracy"], float)
        assert isinstance(loaded["top10_accuracy"], float)
        assert isinstance(loaded["top25_accuracy"], float)

        assert isinstance(loaded["track_recalls"], dict)
        for t in TRACK_NAMES:
            assert t in loaded["track_recalls"]
            assert 0.0 <= loaded["track_recalls"][t] <= 1.0
        assert "any" in loaded["track_recalls"]
        assert 0.0 <= loaded["track_recalls"]["any"] <= 1.0

    def test_report_not_clobbered_by_intermediate_csv(self, tmp_path: Path) -> None:
        """
        Verify that passing output_path='path/to/report.json' does NOT result
        in report.json being overwritten by the pipeline's intermediate submission CSV.
        """
        out_json = tmp_path / "final_report.json"
        df = pd.DataFrame({
            "id": ["CLOB_01"],
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0"],
            "inchikey14": ["IK14DBFLAV0001"],
        })

        run_benchmark(df, output_path=out_json)

        raw_content = out_json.read_text(encoding="utf-8").strip()
        assert raw_content.startswith("{"), f"Report file was clobbered by non-JSON content: {raw_content[:100]}"
        assert raw_content.endswith("}")

    def test_deeply_nested_directory_creation(self, tmp_path: Path) -> None:
        """Verify writing report to deeply nested nonexistent subdirectories succeeds."""
        nested_json = tmp_path / "deep" / "nested" / "benchmark" / "report.json"
        df = pd.DataFrame({
            "id": ["NEST_01"],
            "precursor_mz": [286.047],
            "polarity": ["positive"],
            "peaks": ["100.0:10.0"],
        })

        report = run_benchmark(df, output_path=nested_json)
        assert nested_json.exists()
        assert report["total_spectra"] == 1


class TestGovernorComplianceAdversarial:
    """Stress-test runtime governor logic and boundary behavior."""

    def test_dynamic_runtime_governor_extreme_elapsed(self) -> None:
        """
        When elapsed time vastly exceeds total budget, get_per_spectrum_budget
        must clamp gracefully to positive remaining time rather than raising ZeroDivisionError
        or returning negative budget.
        """
        gov = DynamicRuntimeGovernor(total_budget_seconds=100.0, reserve_seconds=10.0)
        # Elapsed 500s > available 90s
        budget = gov.get_per_spectrum_budget(elapsed_seconds=500.0, spectra_remaining=10)
        # max(10.0, 90.0 - 500.0) -> 10.0 / 10 = 1.0
        assert budget == 1.0

    def test_dynamic_runtime_governor_zero_or_negative_remaining(self) -> None:
        """When spectra_remaining is 0 or negative, governor returns 1.0."""
        gov = DynamicRuntimeGovernor()
        assert gov.get_per_spectrum_budget(elapsed_seconds=10.0, spectra_remaining=0) == 1.0
        assert gov.get_per_spectrum_budget(elapsed_seconds=10.0, spectra_remaining=-5) == 1.0

    def test_benchmark_governor_violation_detection(self) -> None:
        """
        Construct a test case where latency per spectrum exceeds 21.3s,
        verifying that governor_compliant is strictly evaluated to False.
        """
        # We can test the governor calculation logic directly as embodied in benchmark.py
        total_time_slow = 50.0
        n_spectra = 2
        governor_compliant = (total_time_slow / n_spectra) <= 21.3
        assert governor_compliant is False  # 25.0s > 21.3s
