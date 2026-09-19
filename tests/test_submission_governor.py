import numpy as np
import pandas as pd
import pytest

from src.submission.runtime_governor import DynamicRuntimeGovernor
from src.submission.writer import validate_submission


def test_validate_submission_contract():
    expected_ids = ["ID1", "ID2"]
    valid_cands_1 = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    valid_cands_2 = ";".join([f"IK14PROD{i:06d}" for i in range(25)])
    df_valid = pd.DataFrame({"id": expected_ids, "candidates": [valid_cands_1, valid_cands_2]})

    # Valid submission passes
    assert validate_submission(df_valid, expected_ids) is True

    # 1. Duplicate candidate in row
    dup_cands = ";".join(["IK14TEST000000"] * 25)
    df_dup = pd.DataFrame({"id": expected_ids, "candidates": [dup_cands, valid_cands_2]})
    with pytest.raises(AssertionError, match="duplicate InChIKey14"):
        validate_submission(df_dup, expected_ids)

    # 2. Wrong row count
    df_wrong_count = pd.DataFrame({"id": ["ID1"], "candidates": [valid_cands_1]})
    with pytest.raises(AssertionError, match="Row count mismatch"):
        validate_submission(df_wrong_count, expected_ids)

    # 3. Missing 'id' column
    df_missing_id = pd.DataFrame({"identifier": expected_ids, "candidates": [valid_cands_1, valid_cands_2]})
    with pytest.raises(AssertionError, match="Submission must contain 'id' column"):
        validate_submission(df_missing_id, expected_ids)

    # 4. Missing 'candidates' column
    df_missing_cands = pd.DataFrame({"id": expected_ids, "smiles": [valid_cands_1, valid_cands_2]})
    with pytest.raises(AssertionError, match="Submission must contain 'candidates' column"):
        validate_submission(df_missing_cands, expected_ids)

    # 5. Spectrum IDs do not match expected IDs
    df_mismatched_ids = pd.DataFrame({"id": ["OTHER1", "OTHER2"], "candidates": [valid_cands_1, valid_cands_2]})
    with pytest.raises(AssertionError, match="Spectrum IDs do not match expected test set IDs"):
        validate_submission(df_mismatched_ids, expected_ids)

    # 6. Candidate count != 25 (e.g. 24)
    short_cands = ";".join([f"IK14TEST{i:06d}" for i in range(24)])
    df_short = pd.DataFrame({"id": expected_ids, "candidates": [short_cands, valid_cands_2]})
    with pytest.raises(AssertionError, match="expected exactly 25"):
        validate_submission(df_short, expected_ids)

    # 7. Invalid length token (< 14 chars)
    bad_len_cands = ";".join([f"IK14TEST{i:06d}" for i in range(24)] + ["SHORT"])
    df_bad_len = pd.DataFrame({"id": expected_ids, "candidates": [bad_len_cands, valid_cands_2]})
    with pytest.raises(AssertionError, match="invalid InChIKey14 string"):
        validate_submission(df_bad_len, expected_ids)

    # 8. Non-alphanumeric token (e.g. containing hyphen or underscore)
    non_alnum_cands = ";".join([f"IK14TEST{i:06d}" for i in range(24)] + ["IK14-TEST-0001"])
    df_non_alnum = pd.DataFrame({"id": expected_ids, "candidates": [non_alnum_cands, valid_cands_2]})
    with pytest.raises(AssertionError, match="invalid InChIKey14 string"):
        validate_submission(df_non_alnum, expected_ids)

    # 9. NaN string token
    nan_str_cands = ";".join([f"IK14TEST{i:06d}" for i in range(24)] + ["NaN"])
    df_nan_str = pd.DataFrame({"id": expected_ids, "candidates": [nan_str_cands, valid_cands_2]})
    with pytest.raises(AssertionError, match="contains empty or NaN candidate"):
        validate_submission(df_nan_str, expected_ids)

    # 10. Actual NaN/null value in candidates
    df_nan_val = pd.DataFrame({"id": expected_ids, "candidates": [np.nan, valid_cands_2]})
    with pytest.raises(AssertionError, match="contains empty or NaN candidate"):
        validate_submission(df_nan_val, expected_ids)


def test_dynamic_runtime_governor_budget():
    governor = DynamicRuntimeGovernor(total_budget_seconds=32400.0, reserve_seconds=480.0)
    # At start with 1500 spectra remaining
    budget = governor.get_per_spectrum_budget(elapsed_seconds=0.0, spectra_remaining=1500)
    assert 21.2 <= budget <= 21.4


def test_dynamic_runtime_governor_elapsed_and_guards():
    governor = DynamicRuntimeGovernor(total_budget_seconds=32400.0, reserve_seconds=480.0)

    # Available time = 31920.0
    # After 10000s with 1000 remaining -> (31920 - 10000) / 1000 = 21.92s
    budget_mid = governor.get_per_spectrum_budget(elapsed_seconds=10000.0, spectra_remaining=1000)
    assert pytest.approx(budget_mid, rel=1e-3) == 21.92

    # After running past budget, safety floor (10.0s / remaining) kicks in
    budget_overtime = governor.get_per_spectrum_budget(elapsed_seconds=32000.0, spectra_remaining=5)
    assert pytest.approx(budget_overtime, rel=1e-3) == 2.0  # 10.0 / 5

    # Zero or negative remaining spectra guard
    assert governor.get_per_spectrum_budget(elapsed_seconds=100.0, spectra_remaining=0) == 1.0
    assert governor.get_per_spectrum_budget(elapsed_seconds=100.0, spectra_remaining=-10) == 1.0


def test_empty_dataframe_handling():
    empty_df = pd.DataFrame({"id": [], "candidates": []})
    # Empty DataFrame with empty expected IDs is structurally valid
    assert validate_submission(empty_df, []) is True

    # Empty DataFrame with non-empty expected IDs should fail row count check
    with pytest.raises(AssertionError, match="Row count mismatch"):
        validate_submission(empty_df, ["ID1"])
