import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import numpy.testing as npt

from src.submission.writer import (
    DEFAULT_FALLBACK_SMILES,
    IncompleteSubmissionError,
    SubmissionValidationError,
    get_inchikey14,
    validate_submission,
    validate_submission_file,
    write_submission,
)
from src.submission.runtime_governor import (
    BudgetExhaustedError,
    DynamicRuntimeGovernor,
    OperatingRegime,
    RuntimeGovernor,
)


class DummyCandidate:
    def __init__(self, smiles):
        self.smiles = smiles


# ---------------------------------------------------------------------------
# Submission Writer tests (~22)
# ---------------------------------------------------------------------------

def test_write_submission_empty_predictions_backfill(tmp_path):
    out_file = tmp_path / "submission.csv"
    predictions = {}
    expected_ids = ["mol1"]
    
    res = write_submission(
        predictions=predictions,
        output_path=out_file,
        expected_ids=expected_ids,
        auto_backfill=True,
    )
    
    assert res == out_file
    assert out_file.exists()
    
    with open(out_file, "r") as f:
        lines = f.readlines()
        
    assert len(lines) == 2
    assert lines[0].strip() == "molecule_id,smiles"
    
    mol_id, smiles_str = lines[1].strip().split(",")
    assert mol_id == "mol1"
    
    cands = smiles_str.split(";")
    assert len(cands) == 25
    assert cands == DEFAULT_FALLBACK_SMILES[:25]


def test_write_submission_truncate(tmp_path):
    out_file = tmp_path / "submission.csv"
    predictions = {"mol1": ["C" * i for i in range(1, 30)]}
    
    write_submission(
        predictions=predictions,
        output_path=out_file,
        validate_inchikey14=False,
        auto_truncate=True,
    )
    
    with open(out_file, "r") as f:
        lines = f.readlines()
        
    cands = lines[1].strip().split(",")[1].split(";")
    assert len(cands) == 25
    assert cands[0] == "C"
    assert cands[24] == "C" * 25


def test_write_submission_auto_backfill_false_raises(tmp_path):
    out_file = tmp_path / "submission.csv"
    predictions = {"mol1": ["C", "CC"]}
    
    with pytest.raises(SubmissionValidationError, match="valid candidates"):
        write_submission(
            predictions=predictions,
            output_path=out_file,
            auto_backfill=False,
        )


def test_write_submission_auto_truncate_false_raises(tmp_path):
    out_file = tmp_path / "submission.csv"
    predictions = {"mol1": ["C" * i for i in range(1, 30)]}
    
    with pytest.raises(SubmissionValidationError, match="candidates"):
        write_submission(
            predictions=predictions,
            output_path=out_file,
            validate_inchikey14=False,
            auto_truncate=False,
        )


def test_write_submission_atomic_cleanup(tmp_path):
    out_file = tmp_path / "sub.csv"
    predictions = {"mol1": ["C"]}
    
    with patch("os.replace", side_effect=OSError("Boom")):
        with pytest.raises(OSError, match="Boom"):
            write_submission(predictions, out_file, auto_backfill=True)
            
    tmps = list(tmp_path.glob(".*.tmp"))
    assert len(tmps) == 0


def test_write_submission_candidate_objects(tmp_path):
    out_file = tmp_path / "sub.csv"
    predictions = {"mol1": [DummyCandidate("C"), DummyCandidate("CC")]}
    
    write_submission(predictions, out_file, auto_backfill=True)
    
    with open(out_file, "r") as f:
        lines = f.readlines()
        
    cands = lines[1].strip().split(",")[1].split(";")
    assert cands[0] == "C"
    assert cands[1] == "CC"


def test_output_csv_header(tmp_path):
    out_file = tmp_path / "sub.csv"
    write_submission({"m": []}, out_file, auto_backfill=True)
    with open(out_file, "r") as f:
        assert f.readline().strip() == "molecule_id,smiles"


def test_output_csv_entries(tmp_path):
    out_file = tmp_path / "sub.csv"
    write_submission({"m": []}, out_file, auto_backfill=True)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert len(lines[1].strip().split(",")[1].split(";")) == 25


def test_get_inchikey14_valid():
    pytest.importorskip("rdkit")
    ik = get_inchikey14("CC")
    assert isinstance(ik, str)
    assert len(ik) == 14


def test_get_inchikey14_invalid_hash():
    ik = get_inchikey14("INVALID_SMILES_!@#")
    assert isinstance(ik, str)
    assert len(ik) == 14
    assert ik.isupper()
    assert ik.isalnum()


def test_get_inchikey14_none():
    assert get_inchikey14(None) is None


def test_get_inchikey14_empty():
    assert get_inchikey14("") is None


def test_validate_submission_file_valid(tmp_path):
    out_file = tmp_path / "sub.csv"
    write_submission({"m": []}, out_file, auto_backfill=True)
    is_valid, errs = validate_submission_file(out_file)
    assert is_valid
    assert len(errs) == 0


def test_validate_submission_file_missing(tmp_path):
    is_valid, errs = validate_submission_file(tmp_path / "nonexistent.csv")
    assert not is_valid
    assert "does not exist" in errs[0]


def test_validate_submission_file_empty(tmp_path):
    out_file = tmp_path / "empty.csv"
    out_file.touch()
    is_valid, errs = validate_submission_file(out_file)
    assert not is_valid
    assert "0 bytes" in errs[0]


def test_validate_submission_file_wrong_header(tmp_path):
    out_file = tmp_path / "bad.csv"
    out_file.write_text("id,smiles\nm,C")
    is_valid, errs = validate_submission_file(out_file)
    assert not is_valid
    assert any("Invalid CSV header" in e for e in errs)


def test_validate_submission_file_less_than_25(tmp_path):
    out_file = tmp_path / "bad.csv"
    out_file.write_text("molecule_id,smiles\nm,C;CC")
    is_valid, errs = validate_submission_file(out_file)
    assert not is_valid
    assert any("Expected exactly 25" in e for e in errs)


def test_validate_submission_file_duplicate_ids(tmp_path):
    out_file = tmp_path / "bad.csv"
    cands = ";".join(DEFAULT_FALLBACK_SMILES[:25])
    out_file.write_text(f"molecule_id,smiles\nm,{cands}\nm,{cands}")
    is_valid, errs = validate_submission_file(out_file)
    assert not is_valid
    assert any("Duplicate molecule_ids" in e for e in errs)


def test_validate_submission_file_missing_expected(tmp_path):
    out_file = tmp_path / "sub.csv"
    write_submission({"m": []}, out_file, auto_backfill=True)
    is_valid, errs = validate_submission_file(out_file, expected_ids=["m", "missing"])
    assert not is_valid
    assert any("Missing 1 expected" in e for e in errs)


def test_validate_submission_missing_id():
    df = pd.DataFrame({"candidates": ["C"]})
    with pytest.raises(AssertionError, match="must contain 'id'"):
        validate_submission(df)


def test_validate_submission_nan_candidates():
    df = pd.DataFrame({"id": ["m1"], "candidates": [float("nan")]})
    with pytest.raises(AssertionError, match="empty or NaN"):
        validate_submission(df)


def test_fallback_smiles_valid_inchikey14():
    for smiles in DEFAULT_FALLBACK_SMILES:
        assert get_inchikey14(smiles) is not None


# ---------------------------------------------------------------------------
# Runtime Governor tests (~23)
# ---------------------------------------------------------------------------

def test_fresh_governor_regime():
    gov = RuntimeGovernor(total_budget_sec=28800, total_queries=400)
    assert gov.get_execution_mode() == OperatingRegime.DEEP


def test_regime_transitions():
    gov = RuntimeGovernor(total_budget_sec=28800, safety_buffer_sec=0, total_queries=400)
    
    # 28800 / 400 = 72s > 5s -> DEEP
    assert gov.get_execution_mode() == OperatingRegime.DEEP
    
    # Fake time elapsed such that remaining time is 1200s, queries = 400
    # 1200 / 400 = 3s -> STANDARD
    with patch("time.time", return_value=gov.start_time + 27600):
        assert gov.get_execution_mode() == OperatingRegime.STANDARD

    # Fake time elapsed such that remaining time is 400s, queries = 400
    # 400 / 400 = 1s -> FAST
    with patch("time.time", return_value=gov.start_time + 28400):
        assert gov.get_execution_mode() == OperatingRegime.FAST


def test_sec_per_query_remaining():
    gov = RuntimeGovernor(total_budget_sec=1000, safety_buffer_sec=0, total_queries=100)
    # Start: 1000 / 100 = 10.0
    assert gov.sec_per_query_remaining == pytest.approx(10.0, 0.1)
    
    gov.completed_queries = 50
    # 50 remaining queries
    assert gov.sec_per_query_remaining == pytest.approx(20.0, 0.1)


def test_remaining_queries_never_zero():
    gov = RuntimeGovernor(total_budget_sec=1000, total_queries=100)
    gov.completed_queries = 150
    assert gov.remaining_queries == 1


def test_bypass_threshold_exact():
    gov = RuntimeGovernor(bypass_similarity_threshold=0.82)
    assert gov.should_bypass_generative(0.82) is True


def test_bypass_threshold_below():
    gov = RuntimeGovernor(bypass_similarity_threshold=0.82)
    assert gov.should_bypass_generative(0.819) is False


def test_query_scope_telemetry():
    gov = RuntimeGovernor()
    with gov.query_scope("mol1") as regime:
        assert regime == OperatingRegime.DEEP
        gov._current_module_durations["test"] = 1.0
        
    assert len(gov.telemetry_records) == 1
    assert gov.telemetry_records[0].molecule_id == "mol1"
    assert gov.telemetry_records[0].module_durations["test"] == 1.0


def test_module_scope_duration():
    gov = RuntimeGovernor()
    gov.start_query("mol1")
    with gov.module_scope("tier1"):
        pass
    assert "tier1" in gov._current_module_durations
    assert gov._current_module_durations["tier1"] >= 0.0


def test_get_module_timeout_fractions():
    gov = RuntimeGovernor(total_budget_sec=4000, safety_buffer_sec=0, total_queries=400)
    # 4000/400 = 10s per query -> DEEP
    # tier1 in DEEP is 0.05. Cap is 10.0s. 0.05 * 10.0 = 0.5s
    timeout = gov.get_module_timeout("tier1")
    assert timeout == pytest.approx(0.5, 0.001)


def test_fast_regime_allocations():
    gov = RuntimeGovernor(total_budget_sec=400, safety_buffer_sec=0, total_queries=400)
    # 400/400 = 1s per query -> FAST
    # tier3 in FAST is 0.0, cap is 1.0. 0.0 -> max(0.01, 0.0) -> 0.01
    assert gov.get_module_timeout("tier3") == 0.01
    assert gov.get_module_timeout("knapsack") == 0.01


def test_get_module_timeout_unlisted():
    gov = RuntimeGovernor(total_budget_sec=4000, safety_buffer_sec=0, total_queries=400)
    # Unlisted gets 0.10. Cap is 10. 10 * 0.1 = 1.0
    assert gov.get_module_timeout("unknown") == pytest.approx(1.0, 0.001)


def test_export_telemetry_valid_json(tmp_path):
    gov = RuntimeGovernor()
    gov.start_query("m1")
    gov.finish_query()
    
    out_file = tmp_path / "telemetry.json"
    gov.export_telemetry(out_file)
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert "summary" in data
    assert "records" in data
    assert len(data["records"]) == 1


def test_get_summary_keys():
    gov = RuntimeGovernor()
    s = gov.get_summary()
    expected_keys = [
        "total_budget_sec", "safety_buffer_sec", "effective_budget_sec",
        "elapsed_time_sec", "remaining_time_sec", "completed_queries",
        "total_queries", "avg_query_duration_sec", "min_query_duration_sec",
        "max_query_duration_sec", "current_regime", "regime_distribution",
        "bypassed_deep_count"
    ]
    for k in expected_keys:
        assert k in s


def test_budget_exhausted_error_type():
    assert issubclass(BudgetExhaustedError, RuntimeError)


def test_is_budget_exhausted():
    gov = RuntimeGovernor(total_budget_sec=100, safety_buffer_sec=0)
    assert not gov.is_budget_exhausted
    with patch("time.time", return_value=gov.start_time + 150):
        assert gov.is_budget_exhausted


def test_init_zero_budget_raises():
    with pytest.raises(ValueError, match="total_budget_sec must be positive"):
        RuntimeGovernor(total_budget_sec=0)


def test_init_zero_queries_raises():
    with pytest.raises(ValueError, match="total_queries must be positive"):
        RuntimeGovernor(total_queries=0)


def test_start_finish_query_manual():
    gov = RuntimeGovernor()
    regime = gov.start_query("mol1")
    assert isinstance(regime, OperatingRegime)
    assert gov._current_molecule_id == "mol1"
    
    duration = gov.finish_query()
    assert duration >= 0.0
    assert gov._current_molecule_id is None


def test_finish_query_increments_completed():
    gov = RuntimeGovernor()
    assert gov.completed_queries == 0
    gov.start_query("m1")
    gov.finish_query()
    assert gov.completed_queries == 1


def test_dynamic_governor_zero_spectra():
    gov = DynamicRuntimeGovernor()
    assert gov.get_per_spectrum_budget(100.0, 0) == 1.0


def test_dynamic_governor_normal():
    gov = DynamicRuntimeGovernor(total_budget_seconds=1000, reserve_seconds=100)
    # Available = 900
    # Elapsed = 100, remaining = 800
    # Spectra remaining = 8
    # Budget = 800 / 8 = 100.0
    assert gov.get_per_spectrum_budget(100.0, 8) == 100.0


def test_operating_regime_enum():
    assert OperatingRegime.DEEP.value == "DEEP"
    assert OperatingRegime.STANDARD.value == "STANDARD"
    assert OperatingRegime.FAST.value == "FAST"


def test_elapsed_time_non_negative():
    gov = RuntimeGovernor()
    with patch("time.time", return_value=gov.start_time - 100):
        assert gov.elapsed_time == 0.0
