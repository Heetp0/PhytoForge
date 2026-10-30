"""
Unit tests for telemetry and progress logging utilities in src/submission/telemetry.py.
"""

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.submission.telemetry import (
    _PAGE_SIZE,
    format_banner,
    format_completion,
    format_heartbeat,
    get_process_memory_gb,
)


def test_get_process_memory_gb_proc_statm():
    """Verify get_process_memory_gb reads index 1 of /proc/self/statm as RSS pages."""
    fake_statm = "2048 1024 512 100 0 200 0\n"
    with patch("builtins.open", return_value=StringIO(fake_statm)):
        mem_gb = get_process_memory_gb()
        expected_gb = float((1024 * _PAGE_SIZE) / (1024.0 ** 3))
        assert abs(mem_gb - expected_gb) < 1e-9


def test_get_process_memory_gb_fallback_on_os_error():
    """Verify fallback to psutil when /proc/self/statm raises OSError, and 0.0 when psutil fails."""
    # 1. Fallback to psutil
    mock_mem_info = MagicMock()
    mock_mem_info.rss = 2 * (1024 ** 3)  # 2 GB
    mock_process = MagicMock()
    mock_process.memory_info.return_value = mock_mem_info

    with patch("builtins.open", side_effect=OSError("No /proc")):
        with patch("psutil.Process", return_value=mock_process):
            mem_gb = get_process_memory_gb()
            assert abs(mem_gb - 2.0) < 1e-6

    # 2. Complete failure fallback to 0.0
    with patch("builtins.open", side_effect=OSError("No /proc")):
        with patch("psutil.Process", side_effect=RuntimeError("psutil error")):
            mem_gb = get_process_memory_gb()
            assert mem_gb == 0.0


def test_get_process_memory_gb_malformed_statm():
    """Verify malformed /proc/self/statm falls back gracefully."""
    mock_mem_info = MagicMock()
    mock_mem_info.rss = 1073741824  # 1 GB
    mock_process = MagicMock()
    mock_process.memory_info.return_value = mock_mem_info

    with patch("builtins.open", return_value=StringIO("invalid")):
        with patch("psutil.Process", return_value=mock_process):
            mem_gb = get_process_memory_gb()
            assert abs(mem_gb - 1.0) < 1e-6


def test_format_heartbeat():
    """Verify format_heartbeat produces the exact expected structure for various states."""
    # Test 1: Rank 1 match, with adduct, short ETA (< 60s)
    hb1 = format_heartbeat(
        idx=5,
        total=100,
        query_id="spec_042",
        top_cosine=0.8912,
        matched_adduct="[M+H]+",
        is_rank1=True,
        hits_count=4,
        sec_per_spec=1.25,
        ram_gb=3.5,
        eta_sec=45.0,
    )
    expected_hb1 = (
        "[005/100] (  5.0%) Query: spec_042 | Top Cos: 0.8912 ([M+H]+) | "
        "Match: Rank 1 | Hits: 4/5 (80.0%) | 1.25s/mol | RAM: 3.50 GB | ETA: 45s"
    )
    assert hb1 == expected_hb1

    # Test 2: Heuristic match, no adduct, long ETA (>= 60s)
    hb2 = format_heartbeat(
        idx=10,
        total=10,
        query_id="spec_099",
        top_cosine=0.65,
        matched_adduct="",
        is_rank1=False,
        hits_count=0,
        sec_per_spec=2.0,
        ram_gb=4.12,
        eta_sec=150.0,
    )
    expected_hb2 = (
        "[010/010] (100.0%) Query: spec_099 | Top Cos: 0.6500 | "
        "Match: DB/Heuristic | Hits: 0/10 (0.0%) | 2.00s/mol | RAM: 4.12 GB | ETA: 2.5m"
    )
    assert hb2 == expected_hb2

    # Test 3: Zero idx and total edge case (no ZeroDivisionError)
    hb3 = format_heartbeat(
        idx=0,
        total=0,
        query_id="spec_000",
        top_cosine=0.0,
        matched_adduct="",
        is_rank1=False,
        hits_count=0,
        sec_per_spec=0.0,
        ram_gb=1.0,
        eta_sec=-10.0,
    )
    expected_hb3 = (
        "[000/000] (  0.0%) Query: spec_000 | Top Cos: 0.0000 | "
        "Match: DB/Heuristic | Hits: 0/0 (0.0%) | 0.00s/mol | RAM: 1.00 GB | ETA: 0s"
    )
    assert hb3 == expected_hb3


def test_format_banner():
    """Verify format_banner produces header with PhytoForge v1.0.6, assets, and target count."""
    assets = {
        "candidate_db": Path("/kaggle/input/candidates.db"),
        "spectral_index": "dreams_index",
        "meta_ranker": None,
    }
    banner = format_banner(assets=assets, total_queries=150)
    assert "PhytoForge v1.0.6 [Live Telemetry Active]" in banner
    assert "Target Queries: 150" in banner
    assert "candidate_db: candidates.db" in banner
    assert "spectral_index: dreams_index" in banner
    assert "meta_ranker: none" in banner


def test_format_completion():
    """Verify format_completion produces formatted summary of execution statistics."""
    summary = format_completion(
        total_queries=200,
        elapsed_sec=300.0,
        authentic_hits=160,
        peak_ram_gb=5.25,
        output_path=Path("submission.csv"),
    )
    assert "PhytoForge Pipeline Run Completed Successfully" in summary
    assert "Total Queries:   200" in summary
    assert "Total Time:      300.00s (5.00 min)" in summary
    assert "Average Speed:   1.50s/spec" in summary
    assert "Authentic Hits:  160/200 (80.0%)" in summary
    assert "Peak Memory:     5.25 GB" in summary
    assert "Submission File: submission.csv" in summary

    # Edge case: 0 queries
    summary_zero = format_completion(
        total_queries=0,
        elapsed_sec=0.0,
        authentic_hits=0,
        peak_ram_gb=1.0,
        output_path="submission.csv",
    )
    assert "Total Queries:   0" in summary_zero
    assert "Authentic Hits:  0/0 (0.0%)" in summary_zero


def test_get_process_memory_gb_live():
    """Verify live call returns a valid non-negative float."""
    rss_gb = get_process_memory_gb()
    assert isinstance(rss_gb, float)
    assert rss_gb >= 0.0

