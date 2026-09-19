import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from src.pipeline import CASMIOmegaPipeline, run_casmi_omega_pipeline


def test_e2e_pipeline_generates_valid_submission():
    pipeline = CASMIOmegaPipeline()
    # Mock test set with 2 queries
    test_df = pd.DataFrame({
        "id": ["TEST_001", "TEST_002"],
        "precursor_mz": [301.071, 447.129],
        "adduct": ["[M+H]+", "[M+H]+"],
        "polarity": ["positive", "positive"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "submission.csv"
        result_df = pipeline.run(test_df, output_path=out_csv)
        assert out_csv.exists()
        assert len(result_df) == 2
        for _, row in result_df.iterrows():
            cands = row["candidates"].split(";")
            assert len(cands) == 25
            assert len(set(cands)) == 25
            assert all(len(c) == 14 and c.isalnum() for c in cands)
            assert "IK14DREAM00001" in cands or "IK14DBFLAV0001" in cands


def test_e2e_pipeline_empty_dataframe():
    pipeline = CASMIOmegaPipeline()
    empty_df = pd.DataFrame(columns=["id", "precursor_mz", "adduct", "polarity"])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "empty_submission.csv"
        result_df = pipeline.run(empty_df, output_path=out_csv)
        assert out_csv.exists()
        assert len(result_df) == 0
        assert list(result_df.columns) == ["id", "candidates"]


def test_e2e_pipeline_multi_query_processing():
    pipeline = CASMIOmegaPipeline()
    # Multi-query test set with 5 queries
    test_df = pd.DataFrame({
        "id": [f"TEST_{i:03d}" for i in range(1, 6)],
        "precursor_mz": [301.071, 447.129, 287.055, 315.086, 329.102],
        "adduct": ["[M+H]+"] * 5,
        "polarity": ["positive"] * 5,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "multi_submission.csv"
        result_df = run_casmi_omega_pipeline(test_df, output_path=out_csv)
        assert out_csv.exists()
        assert len(result_df) == 5
        assert list(result_df["id"]) == [f"TEST_{i:03d}" for i in range(1, 6)]
        for _, row in result_df.iterrows():
            cands = row["candidates"].split(";")
            assert len(cands) == 25
            assert len(set(cands)) == 25
            assert all(len(c) == 14 and c.isalnum() for c in cands)


def test_e2e_pipeline_validates_before_writing():
    pipeline = CASMIOmegaPipeline()
    test_df = pd.DataFrame({
        "id": ["TEST_001"],
        "precursor_mz": [301.071],
        "adduct": ["[M+H]+"],
        "polarity": ["positive"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "submission.csv"
        with patch("src.pipeline.validate_submission", side_effect=AssertionError("Validation failed!")) as mock_val:
            with pytest.raises(AssertionError, match="Validation failed!"):
                pipeline.run(test_df, output_path=out_csv)
            assert mock_val.called
            # Ensure file is not created if validation fails
            assert not out_csv.exists()
