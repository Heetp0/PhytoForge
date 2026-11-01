"""
tests/test_pipeline_telemetry.py

Integration tests for live telemetry in CASMIOmegaPipeline and run_kaggle_pipeline.
Verifies query heartbeats, completion summaries, and startup banner formatting.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from src.pipeline import CASMIOmegaPipeline
from src.submission.kaggle_bundle import run_kaggle_pipeline


@pytest.fixture
def dummy_3_queries_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["MOL_0001", "MOL_0002", "MOL_0003"],
        "precursor_mz": [195.0877, 205.1234, 215.3333],
        "adduct": ["[M+H]+", "[M+Na]+", "[M-H]-"],
    })


def test_pipeline_telemetry_output(dummy_3_queries_df: pd.DataFrame, tmp_path: Path) -> None:
    """
    Verify CASMIOmegaPipeline.run emits live query heartbeats and completion summary.
    """
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "submission.csv"

    captured_out = io.StringIO()
    with patch("sys.stdout", captured_out):
        pipeline.run(dummy_3_queries_df, output_path=out_file)

    stdout_content = captured_out.getvalue()

    # Heartbeat verification
    assert "[001/003]" in stdout_content
    assert "[003/003]" in stdout_content
    assert "Query: MOL_0001" in stdout_content
    assert "RAM:" in stdout_content
    assert "s/mol" in stdout_content

    # Completion block verification
    assert "Pipeline Run Completed Successfully" in stdout_content
    assert "Total Queries:   3" in stdout_content
    assert f"Submission File: {out_file}" in stdout_content
    assert "Peak Memory:" in stdout_content
    assert "Average Speed:" in stdout_content


def test_pipeline_telemetry_in_memory_run(dummy_3_queries_df: pd.DataFrame) -> None:
    """
    Verify CASMIOmegaPipeline.run handles output_path=None gracefully and logs 'in-memory'.
    """
    pipeline = CASMIOmegaPipeline()
    captured_out = io.StringIO()
    with patch("sys.stdout", captured_out):
        res = pipeline.run(dummy_3_queries_df, output_path=None)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3

    stdout_content = captured_out.getvalue()
    assert "[001/003]" in stdout_content
    assert "Pipeline Run Completed Successfully" in stdout_content
    assert "Submission File: in-memory" in stdout_content


def test_kaggle_bundle_banner_telemetry(tmp_path: Path) -> None:
    """
    Verify run_kaggle_pipeline emits format_banner at startup and completes successfully.
    """
    dummy_df = pd.DataFrame({
        "id": ["MOL_0001", "MOL_0002"],
        "precursor_mz": [195.0877, 205.1234],
        "adduct": ["[M+H]+", "[M+Na]+"],
    })
    test_parquet = tmp_path / "test.parquet"
    dummy_df.to_parquet(test_parquet, index=False)
    out_file = tmp_path / "kaggle_submission.csv"

    captured_out = io.StringIO()
    with patch("sys.stdout", captured_out):
        resolved_out = run_kaggle_pipeline(
            input_path=test_parquet,
            output_path=out_file,
            output_format="inchikey14",
            force_kaggle=False,
        )

    assert resolved_out.is_file()

    stdout_content = captured_out.getvalue()

    # format_banner verification
    assert "PhytoForge v1.0.6 [Live Telemetry Active] - Initializing Pipeline" in stdout_content
    assert "Target Queries: 2" in stdout_content
    assert "Resolved Assets:" in stdout_content
    assert "- DB:" in stdout_content
    assert "- Index:" in stdout_content
    assert "- FragLib:" in stdout_content
    assert "- Ranker:" in stdout_content
    assert "- TrainRef:" in stdout_content

    # Pipeline heartbeats and completion verification
    assert "[001/002]" in stdout_content
    assert "[002/002]" in stdout_content
    assert "Pipeline Run Completed Successfully" in stdout_content
