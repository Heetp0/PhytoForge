import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    CASMIOmegaPipeline,
    PhytoForgePipeline,
    run_casmi_omega_pipeline,
    run_phytoforge_pipeline,
)
from src.submission.writer import (
    DEFAULT_FALLBACK_SMILES,
    IncompleteSubmissionError,
    SubmissionValidationError,
    validate_submission,
    validate_submission_file,
    write_submission,
)


@pytest.fixture
def sample_test_df():
    return pd.DataFrame({
        "id": ["TEST_001", "TEST_002"],
        "precursor_mz": [301.071, 447.129],
        "adduct": ["[M+H]+", "[M+H]+"],
        "polarity": ["positive", "positive"],
    })


# ---------------------------------------------------------------------------
# Unit tests for write_submission
# ---------------------------------------------------------------------------

def test_write_submission_inchikey14_from_df(tmp_path):
    out_file = tmp_path / "submission_ik14.csv"
    cands_1 = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    cands_2 = ";".join([f"IK14PROD{i:06d}" for i in range(25)])
    df = pd.DataFrame({"id": ["MOL_1", "MOL_2"], "candidates": [cands_1, cands_2]})

    res = write_submission(df, out_file, output_format="inchikey14")
    assert res == out_file
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert lines[0] == "id,candidates"
    assert len(lines) == 3
    assert lines[1].startswith("MOL_1,")
    assert len(lines[1].split(",")[1].split(";")) == 25


def test_write_submission_smiles_from_df(tmp_path):
    out_file = tmp_path / "submission_smiles.csv"
    smi_1 = ";".join(DEFAULT_FALLBACK_SMILES[:25])
    smi_2 = ";".join(DEFAULT_FALLBACK_SMILES[5:30])
    df = pd.DataFrame({"molecule_id": ["MOL_1", "MOL_2"], "smiles": [smi_1, smi_2]})

    res = write_submission(df, out_file, output_format="smiles")
    assert res == out_file
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert lines[0] == "molecule_id,smiles"
    assert len(lines) == 3
    assert lines[1].startswith("MOL_1,")
    assert len(lines[1].split(",")[1].split(";")) == 25


def test_write_submission_inchikey14_from_mapping(tmp_path):
    out_file = tmp_path / "sub_map_ik14.csv"
    predictions = {
        "MOL_1": [f"IK14CUSTOM{i:04d}" for i in range(10)],
    }
    write_submission(predictions, out_file, output_format="inchikey14", auto_backfill=True)

    assert out_file.exists()
    is_valid, errors = validate_submission_file(out_file, output_format="inchikey14")
    assert is_valid, f"Validation failed: {errors}"

    df = pd.read_csv(out_file)
    assert list(df.columns) == ["id", "candidates"]
    cands = df.iloc[0]["candidates"].split(";")
    assert len(cands) == 25
    assert len(set(cands)) == 25
    assert all(len(c) == 14 and c.isalnum() for c in cands)


def test_write_submission_invalid_format_raises(tmp_path):
    out_file = tmp_path / "sub_invalid.csv"
    df = pd.DataFrame({"id": ["M1"], "candidates": [";".join([f"IK14TEST{i:06d}" for i in range(25)])]})
    with pytest.raises(ValueError, match="Invalid output_format"):
        write_submission(df, out_file, output_format="unsupported_format")


# ---------------------------------------------------------------------------
# Unit tests for validate_submission
# ---------------------------------------------------------------------------

def test_validate_submission_inchikey14_invariants():
    cands = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    df = pd.DataFrame({"id": ["M1"], "candidates": [cands]})
    assert validate_submission(df, output_format="inchikey14") is True

    # Duplicate InChIKeys
    dup_cands = ";".join(["IK14TEST000000"] * 25)
    df_dup = pd.DataFrame({"id": ["M1"], "candidates": [dup_cands]})
    with pytest.raises(AssertionError, match="duplicate InChIKey14"):
        validate_submission(df_dup, output_format="inchikey14")

    # Invalid length/alnum
    bad_ik = ";".join([f"IK14TEST{i:06d}" for i in range(24)] + ["INVALID_IK_!@#"])
    df_bad = pd.DataFrame({"id": ["M1"], "candidates": [bad_ik]})
    with pytest.raises(AssertionError, match="invalid InChIKey14 string"):
        validate_submission(df_bad, output_format="inchikey14")


def test_validate_submission_smiles_invariants():
    smi = ";".join(DEFAULT_FALLBACK_SMILES[:25])
    df = pd.DataFrame({"molecule_id": ["M1"], "smiles": [smi]})
    assert validate_submission(df, output_format="smiles") is True

    # Missing column
    df_missing = pd.DataFrame({"molecule_id": ["M1"], "other": [smi]})
    with pytest.raises(AssertionError, match="must contain 'smiles' column"):
        validate_submission(df_missing, output_format="smiles")

    # Wrong count
    df_few = pd.DataFrame({"molecule_id": ["M1"], "smiles": ["C;CC;CCC"]})
    with pytest.raises(AssertionError, match="has 3 candidates; expected exactly 25"):
        validate_submission(df_few, output_format="smiles")


# ---------------------------------------------------------------------------
# Unit tests for validate_submission_file
# ---------------------------------------------------------------------------

def test_validate_submission_file_format_checking(tmp_path):
    ik_file = tmp_path / "sub_ik14.csv"
    cands_ik = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    ik_file.write_text(f"id,candidates\nM1,{cands_ik}\n")

    # Correct format passes
    valid, errs = validate_submission_file(ik_file, output_format="inchikey14")
    assert valid and len(errs) == 0

    # Auto-detection passes
    valid_auto, errs_auto = validate_submission_file(ik_file)
    assert valid_auto and len(errs_auto) == 0

    # Mismatched format fails
    valid_mismatch, errs_mismatch = validate_submission_file(ik_file, output_format="smiles")
    assert not valid_mismatch
    assert any("Invalid CSV header" in e for e in errs_mismatch)


# ---------------------------------------------------------------------------
# Pipeline integration tests with dual formats
# ---------------------------------------------------------------------------

def test_pipeline_run_inchikey14_format(sample_test_df, tmp_path):
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "sub_ik14.csv"
    res_df = pipeline.run(sample_test_df, out_file, output_format="inchikey14")

    assert out_file.exists()
    assert list(res_df.columns) == ["id", "candidates"]
    assert len(res_df) == 2

    is_valid, errors = validate_submission_file(out_file, output_format="inchikey14")
    assert is_valid, f"Submission file validation failed: {errors}"


def test_pipeline_run_smiles_format(sample_test_df, tmp_path):
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "sub_smiles.csv"
    res_df = pipeline.run(sample_test_df, out_file, output_format="smiles")

    assert out_file.exists()
    assert list(res_df.columns) == ["molecule_id", "smiles"]
    assert len(res_df) == 2

    for _, row in res_df.iterrows():
        cands = row["smiles"].split(";")
        assert len(cands) == 25
        assert len(set(cands)) == 25

    is_valid, errors = validate_submission_file(out_file, output_format="smiles")
    assert is_valid, f"Submission file validation failed: {errors}"


def test_pipeline_run_empty_df_dual_format(tmp_path):
    pipeline = CASMIOmegaPipeline()
    empty_df = pd.DataFrame(columns=["id", "precursor_mz", "adduct"])

    out_ik = tmp_path / "empty_ik.csv"
    res_ik = pipeline.run(empty_df, out_ik, output_format="inchikey14")
    assert list(res_ik.columns) == ["id", "candidates"]
    assert len(res_ik) == 0
    assert out_ik.exists()

    out_smi = tmp_path / "empty_smi.csv"
    res_smi = pipeline.run(empty_df, out_smi, output_format="smiles")
    assert list(res_smi.columns) == ["molecule_id", "smiles"]
    assert len(res_smi) == 0
    assert out_smi.exists()


def test_run_phytoforge_pipeline_dual_format(sample_test_df, tmp_path):
    out_ik = tmp_path / "phyto_ik.csv"
    res_ik = run_phytoforge_pipeline(sample_test_df, out_ik, output_format="inchikey14")
    assert list(res_ik.columns) == ["id", "candidates"]
    assert out_ik.exists()

    out_smi = tmp_path / "phyto_smi.csv"
    res_smi = run_phytoforge_pipeline(sample_test_df, out_smi, output_format="smiles")
    assert list(res_smi.columns) == ["molecule_id", "smiles"]
    assert out_smi.exists()


# ---------------------------------------------------------------------------
# CLI Runner and Validator End-to-End Integration Tests
# ---------------------------------------------------------------------------

def test_cli_pipeline_and_validator_csv(sample_test_df, tmp_path):
    input_csv = tmp_path / "input.csv"
    sample_test_df.to_csv(input_csv, index=False)

    # 1. inchikey14 format
    out_ik = tmp_path / "cli_out_ik.csv"
    cmd_run_ik = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(input_csv),
        "--output", str(out_ik),
        "--format", "inchikey14",
    ]
    res_run_ik = subprocess.run(cmd_run_ik, capture_output=True, text=True)
    assert res_run_ik.returncode == 0, f"CLI runner failed: {res_run_ik.stderr}"
    assert out_ik.exists()

    cmd_val_ik = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(out_ik),
        "--format", "inchikey14",
    ]
    res_val_ik = subprocess.run(cmd_val_ik, capture_output=True, text=True)
    assert res_val_ik.returncode == 0, f"CLI validator failed: {res_val_ik.stderr}"
    assert "Validation successful" in res_val_ik.stdout

    # 2. smiles format
    out_smi = tmp_path / "cli_out_smi.csv"
    cmd_run_smi = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(input_csv),
        "--output", str(out_smi),
        "--format", "smiles",
    ]
    res_run_smi = subprocess.run(cmd_run_smi, capture_output=True, text=True)
    assert res_run_smi.returncode == 0, f"CLI runner failed: {res_run_smi.stderr}"
    assert out_smi.exists()

    cmd_val_smi = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(out_smi),
        "--format", "smiles",
    ]
    res_val_smi = subprocess.run(cmd_val_smi, capture_output=True, text=True)
    assert res_val_smi.returncode == 0, f"CLI validator failed: {res_val_smi.stderr}"
    assert "Validation successful" in res_val_smi.stdout


def test_cli_pipeline_and_validator_parquet(sample_test_df, tmp_path):
    pytest.importorskip("pyarrow")
    input_parquet = tmp_path / "input.parquet"
    sample_test_df.to_parquet(input_parquet, index=False)

    out_ik = tmp_path / "cli_parquet_ik.csv"
    cmd_run = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(input_parquet),
        "--output", str(out_ik),
        "--format", "inchikey14",
    ]
    res_run = subprocess.run(cmd_run, capture_output=True, text=True)
    assert res_run.returncode == 0, f"CLI runner failed: {res_run.stderr}"
    assert out_ik.exists()

    cmd_val = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(out_ik),
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    assert res_val.returncode == 0, f"CLI validator failed: {res_val.stderr}"
    assert "Validation successful" in res_val.stdout


def test_cli_validator_failure_mismatch(tmp_path):
    cands_ik = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    sub_file = tmp_path / "sub_ik.csv"
    sub_file.write_text(f"id,candidates\nM1,{cands_ik}\n")

    cmd_val = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(sub_file),
        "--format", "smiles",
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    assert res_val.returncode == 1
    assert "Validation failed" in res_val.stderr


# ---------------------------------------------------------------------------
# Adversarial Reviewer Boundary & Invariant Edge Case Tests
# ---------------------------------------------------------------------------

def test_pipeline_run_integer_ids(tmp_path):
    """Verifies pipeline correctly handles integer IDs without set-type mismatches."""
    pipeline = CASMIOmegaPipeline()
    int_id_df = pd.DataFrame({
        "id": [101, 102],
        "precursor_mz": [300.0, 310.0],
    })
    out_file = tmp_path / "sub_int_ids.csv"
    res_df = pipeline.run(int_id_df, out_file, output_format="inchikey14")
    assert out_file.exists()
    assert list(res_df.columns) == ["id", "candidates"]
    assert set(res_df["id"]) == {"101", "102"}

    valid, errors = validate_submission_file(out_file, expected_ids=["101", "102"], output_format="inchikey14")
    assert valid and len(errors) == 0


def test_pipeline_run_non_zero_index_without_ids(tmp_path):
    """Verifies pipeline correctly generates and matches IDs when index is non-zero."""
    pipeline = CASMIOmegaPipeline()
    non_zero_df = pd.DataFrame(
        {"precursor_mz": [300.0, 310.0]},
        index=[5, 10],
    )
    out_file = tmp_path / "sub_nonzero_idx.csv"
    res_df = pipeline.run(non_zero_df, out_file, output_format="inchikey14")
    assert out_file.exists()
    assert list(res_df["id"]) == ["SPEC_0000", "SPEC_0001"]

    valid, errors = validate_submission_file(out_file, expected_ids=["SPEC_0000", "SPEC_0001"])
    assert valid and len(errors) == 0


def test_validate_submission_rejects_duplicate_ids_when_expected_ids_none():
    """Verifies validate_submission catches duplicate IDs even when expected_ids=None."""
    cands = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    df = pd.DataFrame({"id": ["M1", "M1"], "candidates": [cands, cands]})
    with pytest.raises(AssertionError, match="Duplicate id entries"):
        validate_submission(df, output_format="inchikey14")


def test_validate_submission_rejects_duplicate_smiles_within_row():
    """Verifies validate_submission catches duplicate SMILES entries in a row."""
    cands = ";".join(["CCO"] * 25)
    df = pd.DataFrame({"molecule_id": ["M1"], "smiles": [cands]})
    with pytest.raises(AssertionError, match="duplicate SMILES entries"):
        validate_submission(df, output_format="smiles")


def test_validate_submission_file_rejects_duplicate_smiles_without_inchikey_check(tmp_path):
    """Verifies validate_submission_file catches duplicate SMILES strings even with check_inchikey14=False."""
    sub_file = tmp_path / "sub_dup_smi.csv"
    cands = ";".join(["CCO"] * 25)
    sub_file.write_text(f"molecule_id,smiles\nM1,{cands}\n")
    valid, errors = validate_submission_file(sub_file, check_inchikey14=False, output_format="smiles")
    assert not valid
    assert any("Duplicate SMILES" in e for e in errors)


def test_write_submission_dataframe_reordered_and_extra_columns(tmp_path):
    """Verifies write_submission normalizes reversed column order and ignores extra metadata columns."""
    cands = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    df = pd.DataFrame({
        "extra_info": ["meta1", "meta2"],
        "candidates": [cands, cands],
        "id": ["M1", "M2"],
    })
    out_file = tmp_path / "sub_norm_cols.csv"
    write_submission(df, out_file, output_format="inchikey14")
    assert out_file.exists()

    df_out = pd.read_csv(out_file)
    assert list(df_out.columns) == ["id", "candidates"]
    assert len(df_out) == 2


def test_validate_submission_rejects_extra_columns():
    """Verifies validate_submission strictly rejects non-canonical columns."""
    cands = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    df = pd.DataFrame({
        "id": ["M1"],
        "candidates": [cands],
        "extra_meta": ["bad"],
    })
    with pytest.raises(AssertionError, match="must be strictly"):
        validate_submission(df, output_format="inchikey14")


def test_write_submission_dataframe_auto_truncate_and_backfill(tmp_path):
    """Verifies write_submission handles truncation (>25) and backfilling (<25) for DataFrame inputs."""
    c30 = ";".join([f"IK14TEST{i:06d}" for i in range(30)])
    c10 = ";".join([f"IK14TEST{i:06d}" for i in range(10)])
    df = pd.DataFrame({"id": ["M_OVER", "M_UNDER"], "candidates": [c30, c10]})

    out_file = tmp_path / "sub_trunc_backfill.csv"
    write_submission(df, out_file, output_format="inchikey14", auto_truncate=True, auto_backfill=True)
    assert out_file.exists()

    valid, errors = validate_submission_file(out_file, output_format="inchikey14")
    assert valid and len(errors) == 0

    df_out = pd.read_csv(out_file)
    assert len(df_out.iloc[0]["candidates"].split(";")) == 25
    assert len(df_out.iloc[1]["candidates"].split(";")) == 25


def test_write_submission_dataframe_missing_expected_ids_auto_backfill(tmp_path):
    """Verifies write_submission auto-backfills completely missing expected IDs from DataFrame input."""
    c25 = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    df = pd.DataFrame({"id": ["M1"], "candidates": [c25]})

    out_file = tmp_path / "sub_backfill_missing.csv"
    write_submission(df, out_file, expected_ids=["M1", "M2_MISSING"], auto_backfill=True)
    assert out_file.exists()

    valid, errors = validate_submission_file(out_file, expected_ids=["M1", "M2_MISSING"])
    assert valid and len(errors) == 0

    df_out = pd.read_csv(out_file)
    assert list(df_out["id"]) == ["M1", "M2_MISSING"]


def test_cli_validator_nonexistent_expected_ids_fails(tmp_path):
    """Verifies CLI validator fails with exit code 1 if expected-ids file does not exist."""
    cands_ik = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    sub_file = tmp_path / "sub_valid.csv"
    sub_file.write_text(f"id,candidates\nM1,{cands_ik}\n")

    cmd = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(sub_file),
        "--expected-ids", str(tmp_path / "non_existent_file.txt"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "not found" in res.stderr


def test_cli_pipeline_latin1_encoding(tmp_path):
    """Verifies CLI pipeline runner handles Latin-1 encoded CSV input without crashing."""
    latin_csv = tmp_path / "latin1_input.csv"
    content = "id,precursor_mz,compound\nM1,300.0,Café\n"
    latin_csv.write_bytes(content.encode("latin-1"))

    out_csv = tmp_path / "latin1_out.csv"
    cmd = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(latin_csv),
        "--output", str(out_csv),
        "--format", "inchikey14",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Failed on Latin-1 input: {res.stderr}"
    assert out_csv.exists()

    valid, errors = validate_submission_file(out_csv, expected_ids=["M1"], output_format="inchikey14")
    assert valid and len(errors) == 0


def test_cli_pipeline_non_zero_index_parquet(tmp_path):
    """Verifies CLI pipeline runner handles Parquet files with non-zero index and missing ID column."""
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "nonzero_idx.parquet"
    pd.DataFrame({"precursor_mz": [300.0, 310.0]}, index=[50, 100]).to_parquet(parquet_path)

    out_csv = tmp_path / "nonzero_out.csv"
    cmd = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(parquet_path),
        "--output", str(out_csv),
        "--format", "inchikey14",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Failed on non-zero index Parquet: {res.stderr}"
    assert out_csv.exists()

    valid, errors = validate_submission_file(out_csv, expected_ids=["SPEC_0000", "SPEC_0001"], output_format="inchikey14")
    assert valid and len(errors) == 0


# ---------------------------------------------------------------------------
# Review Round 2 Adversarial Edge Case Tests
# ---------------------------------------------------------------------------

def test_cli_format_case_and_whitespace_insensitivity(sample_test_df, tmp_path):
    """Verifies CLI runner and validator accept uppercase and whitespace in --format."""
    input_csv = tmp_path / "cli_input_case.csv"
    sample_test_df.to_csv(input_csv, index=False)

    out_ik = tmp_path / "cli_out_upper_ik.csv"
    cmd_run = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(input_csv),
        "--output", str(out_ik),
        "--format", "  INCHIKEY14  ",
    ]
    res_run = subprocess.run(cmd_run, capture_output=True, text=True)
    assert res_run.returncode == 0, f"Failed with uppercase format: {res_run.stderr}"
    assert out_ik.exists()

    cmd_val = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(out_ik),
        "--format", "INCHIKEY14",
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    assert res_val.returncode == 0, f"Validator failed with uppercase format: {res_val.stderr}"
    assert "Validation successful" in res_val.stdout

    # Test SMILES with whitespace
    out_smi = tmp_path / "cli_out_space_smi.csv"
    cmd_run_smi = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(input_csv),
        "--output", str(out_smi),
        "--format", "  smiles ",
    ]
    res_run_smi = subprocess.run(cmd_run_smi, capture_output=True, text=True)
    assert res_run_smi.returncode == 0, f"Failed with whitespace format: {res_run_smi.stderr}"
    assert out_smi.exists()

    cmd_val_smi = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(out_smi),
        "--format", " SMILES ",
    ]
    res_val_smi = subprocess.run(cmd_val_smi, capture_output=True, text=True)
    assert res_val_smi.returncode == 0, f"Validator failed with mixed case format: {res_val_smi.stderr}"


def test_cli_directory_as_file_fails_cleanly(tmp_path):
    """Verifies CLI pipeline and validator fail cleanly if a directory is passed where a file is required."""
    cmd_run = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(tmp_path),
        "--output", str(tmp_path / "out.csv"),
    ]
    res_run = subprocess.run(cmd_run, capture_output=True, text=True)
    assert res_run.returncode == 1
    assert "not a valid file" in res_run.stderr

    cmd_val = [
        sys.executable, "-m", "src.submission.validator",
        "--submission", str(tmp_path),
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    assert res_val.returncode == 1
    assert "not a valid file" in res_val.stderr


def test_validate_submission_file_expected_ids_row_count_and_duplicate():
    """Verifies validate_submission_file catches duplicate expected IDs and row count mismatch."""
    cands_ik = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tf:
        tf.write(f"id,candidates\nM1,{cands_ik}\n")
        tf_name = tf.name

    try:
        # Row count mismatch (expected 2 rows, file has 1)
        valid, errs = validate_submission_file(tf_name, expected_ids=["M1", "M2"], output_format="inchikey14")
        assert not valid
        assert any("Row count mismatch" in e for e in errs)

        # Duplicate expected IDs
        valid_dup, errs_dup = validate_submission_file(tf_name, expected_ids=["M1", "M1"], output_format="inchikey14")
        assert not valid_dup
        assert any("Duplicate expected IDs" in e for e in errs_dup)
    finally:
        os.remove(tf_name)


def test_validate_submission_rejects_candidate_whitespace_padding():
    """Verifies both validator functions reject candidate slots with whitespace padding."""
    # InChIKey with whitespace padding
    cands_pad = ";".join([" IK14TEST000000 "] + [f"IK14TEST{i:06d}" for i in range(1, 25)])
    df_ik = pd.DataFrame({"id": ["M1"], "candidates": [cands_pad]})
    with pytest.raises(AssertionError, match="leading/trailing whitespace"):
        validate_submission(df_ik, output_format="inchikey14")

    # SMILES with whitespace padding
    smiles_pad = ";".join([" C "] + DEFAULT_FALLBACK_SMILES[1:25])
    df_smi = pd.DataFrame({"molecule_id": ["M1"], "smiles": [smiles_pad]})
    with pytest.raises(AssertionError, match="leading/trailing whitespace"):
        validate_submission(df_smi, output_format="smiles")


def test_validate_submission_rejects_smiles_internal_whitespace():
    """Verifies validator functions reject SMILES strings containing internal spaces or tabs."""
    bad_smiles = ";".join(["C C"] + DEFAULT_FALLBACK_SMILES[1:25])
    df = pd.DataFrame({"molecule_id": ["M1"], "smiles": [bad_smiles]})
    with pytest.raises(AssertionError, match="illegal delimiter/whitespace"):
        validate_submission(df, output_format="smiles")


def test_pipeline_run_handles_nan_and_corrupt_precursor_and_adduct(tmp_path):
    """Verifies pipeline run gracefully recovers from NaN/negative precursor m/z and corrupted/None adduct."""
    pipeline = CASMIOmegaPipeline()
    messy_df = pd.DataFrame({
        "id": ["SPEC_CORRUPT_1", "SPEC_CORRUPT_2", "SPEC_CORRUPT_3"],
        "precursor_mz": [np.nan, -25.0, 450.12],
        "adduct": [None, "NOT_AN_ADDUCT", np.nan],
        "polarity": ["positive", "negative", "positive"],
    })

    out_file = tmp_path / "sub_messy.csv"
    res_df = pipeline.run(messy_df, out_file, output_format="inchikey14")
    assert out_file.exists()
    assert len(res_df) == 3
    assert list(res_df["id"]) == ["SPEC_CORRUPT_1", "SPEC_CORRUPT_2", "SPEC_CORRUPT_3"]

    valid, errors = validate_submission_file(out_file, expected_ids=list(res_df["id"]), output_format="inchikey14")
    assert valid and len(errors) == 0, f"Validation failed: {errors}"


def test_pipeline_run_handles_none_test_df(tmp_path):
    """Verifies pipeline run accepts None test_df without raising AttributeError."""
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "sub_none.csv"
    res = pipeline.run(None, out_file, output_format="smiles")
    assert out_file.exists()
    assert len(res) == 0
    assert list(res.columns) == ["molecule_id", "smiles"]


def test_write_submission_handles_empty_dataframe_and_rejects_single_column(tmp_path):
    """Verifies write_submission handles pd.DataFrame() and rejects 1-column DataFrames cleanly."""
    out_empty = tmp_path / "sub_empty_df.csv"
    res_empty = write_submission(pd.DataFrame(), out_empty, output_format="inchikey14")
    assert out_empty.exists()
    assert res_empty == out_empty

    # Single column DataFrame
    out_1col = tmp_path / "sub_1col.csv"
    df_1col = pd.DataFrame({"id": ["M1"]})
    with pytest.raises(SubmissionValidationError, match="must have at least 2 columns"):
        write_submission(df_1col, out_1col, output_format="inchikey14")


def test_write_submission_rejects_existing_directory(tmp_path):
    """Verifies write_submission raises SubmissionValidationError if target path is a directory."""
    dir_target = tmp_path / "existing_dir"
    dir_target.mkdir()

    with pytest.raises(SubmissionValidationError, match="existing directory"):
        write_submission({"M1": ["IK14TEST000000"]}, dir_target, output_format="inchikey14")


def test_write_submission_smiles_exhaustion_alkane_backfill(tmp_path):
    """Verifies write_submission guarantees 25 unique candidates via alkanes when fallback list is exhausted."""
    # Pre-occupy all 41 default fallbacks so fallback pool is exhausted
    out_file = tmp_path / "sub_alkane_backfill.csv"
    preds = {"M1": DEFAULT_FALLBACK_SMILES[:10]}

    write_submission(preds, out_file, output_format="smiles", auto_backfill=True)
    assert out_file.exists()

    valid, errors = validate_submission_file(out_file, output_format="smiles")
    assert valid and len(errors) == 0

    df = pd.read_csv(out_file)
    cands = df.iloc[0]["smiles"].split(";")
    assert len(cands) == 25
    assert len(set(cands)) == 25


def test_write_submission_preserves_ndarray_candidates(tmp_path):
    """Verifies write_submission properly handles numpy array candidates without dropping them."""
    expected_cands = [f"IK14TEST{i:06d}" for i in range(25)]
    arr = np.array(expected_cands)

    # 1. Via DataFrame
    df = pd.DataFrame({"id": ["M1"], "candidates": [arr]})
    out_df = tmp_path / "sub_arr_df.csv"
    write_submission(df, out_df, output_format="inchikey14")
    assert out_df.exists()

    df_out = pd.read_csv(out_df)
    cands_out = df_out.iloc[0]["candidates"].split(";")
    assert cands_out == expected_cands

    # 2. Via dict
    preds_dict = {"M2": arr}
    out_dict = tmp_path / "sub_arr_dict.csv"
    write_submission(preds_dict, out_dict, output_format="inchikey14")
    assert out_dict.exists()

    dict_out = pd.read_csv(out_dict)
    cands_dict_out = dict_out.iloc[0]["candidates"].split(";")
    assert cands_dict_out == expected_cands


def test_validate_submission_file_rfc4180_quotes(tmp_path):
    """Verifies validate_submission_file correctly parses RFC 4180 quoted CSV entries."""
    cands_ik = ";".join([f"IK14TEST{i:06d}" for i in range(25)])
    sub_file = tmp_path / "sub_quoted.csv"
    sub_file.write_text(f'id,candidates\nM1,"{cands_ik}"\n')

    valid, errors = validate_submission_file(sub_file, output_format="inchikey14")
    assert valid and len(errors) == 0, f"Errors: {errors}"


def test_validate_submission_file_directory_rejected(tmp_path):
    """Verifies validate_submission_file returns False and descriptive error if path is a directory."""
    valid, errors = validate_submission_file(tmp_path)
    assert not valid
    assert any("directory" in e for e in errors)


def test_cli_pipeline_rejects_existing_directory_output(sample_test_df, tmp_path):
    """Verifies CLI pipeline runner fails fast if --output is an existing directory."""
    input_csv = tmp_path / "in.csv"
    sample_test_df.to_csv(input_csv, index=False)

    out_dir = tmp_path / "target_dir"
    out_dir.mkdir()

    cmd = [
        sys.executable, "-m", "src.submission.pipeline",
        "--input", str(input_csv),
        "--output", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "directory, not a file" in res.stderr


def test_write_submission_filters_corrupted_smiles_and_backfills(tmp_path):
    """Verifies write_submission skips candidate SMILES with illegal spaces and backfills cleanly."""
    out_file = tmp_path / "sub_corrupt_smi.csv"
    preds = {"M1": ["C C", "CC\tO", "CCC\n"] + DEFAULT_FALLBACK_SMILES[:22]}

    write_submission(preds, out_file, output_format="smiles", auto_backfill=True)
    assert out_file.exists()

    valid, errors = validate_submission_file(out_file, output_format="smiles")
    assert valid and len(errors) == 0, f"Errors: {errors}"

    df = pd.read_csv(out_file)
    cands = df.iloc[0]["smiles"].split(";")
    assert len(cands) == 25
    assert not any(" " in c or "\t" in c or "\n" in c for c in cands)


def test_validate_submission_handles_array_and_list_candidates():
    """Verifies validate_submission accepts DataFrames where candidate column contains numpy arrays or lists."""
    cands = [f"IK14TEST{i:06d}" for i in range(25)]
    df_arr = pd.DataFrame({"id": ["M1"], "candidates": [np.array(cands)]})
    assert validate_submission(df_arr, output_format="inchikey14") is True

    df_list = pd.DataFrame({"molecule_id": ["M1"], "smiles": [DEFAULT_FALLBACK_SMILES[:25]]})
    assert validate_submission(df_list, output_format="smiles") is True



