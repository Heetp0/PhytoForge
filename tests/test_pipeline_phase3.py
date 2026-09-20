"""
tests/test_pipeline_phase3.py
Comprehensive integration test suite for PhytoForge Phase 3 pipeline:
- CASMIOmegaPipeline context manager lifecycle (close() on __exit__)
- db_path dependency injection with minimal SQLite repository
- fragment_library injection and Track K candidate aggregation
- 5-spectrum end-to-end dry-run completing in < 15s
- submission.csv schema and invariant validation (25 slots, 14-char alphanumeric)
- SMILES output format dry-run and chemistry validity
- Wrapper function execution with context manager
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    CASMIOmegaPipeline,
    PhytoForgePipeline,
    run_casmi_omega_pipeline,
    run_phytoforge_pipeline,
)
from src.retrieval.database_search import SoftDatabaseSearcher
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever
from src.submission.writer import (
    get_inchikey14,
    validate_submission,
    validate_submission_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_sqlite_db(tmp_path: Path) -> Path:
    """Creates a minimal SQLite database matching ChemicalDatabaseRepository schema."""
    db_file = tmp_path / "test_compounds.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE candidates ("
        "formula TEXT, "
        "inchikey14 TEXT, "
        "smiles TEXT, "
        "fingerprint BLOB, "
        "PRIMARY KEY (formula, inchikey14)"
        ") WITHOUT ROWID;"
    )
    conn.execute(
        "CREATE TABLE _schema_metadata (key TEXT PRIMARY KEY, value TEXT);"
    )
    conn.execute(
        "INSERT INTO _schema_metadata VALUES ('schema_version', '1.0');"
    )
    # 32 uint64 words = 256 bytes
    fp_blob = np.zeros(32, dtype=np.uint64).tobytes()
    conn.execute(
        "INSERT INTO candidates VALUES (?, ?, ?, ?)",
        ("C15H10O5", "IK14DBFLAV0001", "c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1", fp_blob),
    )
    conn.execute(
        "INSERT INTO candidates VALUES (?, ?, ?, ?)",
        ("C15H10O6", "IK14DBFLAV0002", "c1cc(O)c2c(=O)cc(-c3ccc(O)cc3)oc2c1", fp_blob),
    )
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def sample_5_spectra_df() -> pd.DataFrame:
    """5 realistic test spectra for E2E dry-run benchmarking."""
    return pd.DataFrame({
        "id": [f"SPEC_P3_{i:03d}" for i in range(5)],
        "precursor_mz": [271.060, 287.055, 301.071, 315.086, 447.129],
        "adduct": ["[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+"],
        "polarity": ["positive", "positive", "positive", "positive", "positive"],
    })


@pytest.fixture
def synthetic_fragment_library() -> Dict[int, List[str]]:
    """Synthetic fragment library keyed by integer millimass units."""
    # Precursor neutral masses for [M+H]+:
    # 271.060 - 1.007276 = 270.0527 -> key ~ 2700527
    # 287.055 - 1.007276 = 286.0477 -> key ~ 2860477
    # 301.071 - 1.007276 = 300.0637 -> key ~ 3000637
    # 315.086 - 1.007276 = 314.0787 -> key ~ 3140787
    # 447.129 - 1.007276 = 446.1217 -> key ~ 4461217
    return {
        2700527: ["c1ccccc1O", "c1cc(O)ccc1O"],
        2860477: ["c1cc(O)c(O)cc1O"],
        3000637: ["c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1"],
        3140787: ["COc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1"],
        4461217: ["c1cc(O)c2c(=O)cc(-c3ccc(O)cc3)oc2c1"],
    }


# ===========================================================================
# 1. Context Manager Lifecycle Tests
# ===========================================================================


class TestPipelineContextManager:
    """Test CASMIOmegaPipeline context manager protocol and resource disposal."""

    def test_context_manager_enter_exit(self):
        with CASMIOmegaPipeline() as pipeline:
            assert isinstance(pipeline, CASMIOmegaPipeline)
            assert hasattr(pipeline, "close")

    def test_close_called_on_exit(self):
        pipeline = CASMIOmegaPipeline()
        with patch.object(pipeline, "close", wraps=pipeline.close) as mock_close:
            with pipeline as p:
                assert p is pipeline
            mock_close.assert_called_once()

    def test_close_delegates_to_searcher_and_retriever(self):
        pipeline = CASMIOmegaPipeline()
        with patch.object(pipeline.db_searcher, "close") as mock_db_close, \
             patch.object(pipeline.dreams_retriever, "close") as mock_dreams_close:
            pipeline.close()
            mock_db_close.assert_called_once()
            mock_dreams_close.assert_called_once()

    def test_context_manager_with_exception_calls_close(self):
        pipeline = CASMIOmegaPipeline()
        with patch.object(pipeline, "close", wraps=pipeline.close) as mock_close:
            with pytest.raises(RuntimeError, match="Test Error"):
                with pipeline:
                    raise RuntimeError("Test Error")
            mock_close.assert_called_once()


# ===========================================================================
# 2. Dependency Injection Tests (db_path & popcount_tanimoto handling)
# ===========================================================================


class TestDatabaseDependencyInjection:
    """Test db_path injection into CASMIOmegaPipeline and popcount query compatibility."""

    def test_db_path_injected_activates_repo(self, minimal_sqlite_db: Path):
        with CASMIOmegaPipeline(db_path=minimal_sqlite_db) as pipeline:
            assert pipeline.db_path == minimal_sqlite_db
            assert pipeline.db_searcher.repo is not None

    def test_db_search_with_sqlite_repo_does_not_shape_mismatch(
        self, minimal_sqlite_db: Path, tmp_path: Path
    ):
        """Verify that 32-word uint64 query fingerprint avoids popcount_tanimoto ValueError."""
        test_df = pd.DataFrame({
            "id": ["TEST_DB_001"],
            "precursor_mz": [301.071],
            "adduct": ["[M+H]+"],
            "polarity": ["positive"],
        })
        out_csv = tmp_path / "sub_db_test.csv"

        with CASMIOmegaPipeline(db_path=minimal_sqlite_db) as pipeline:
            res_df = pipeline.run(test_df, out_csv, output_format="inchikey14")

        assert len(res_df) == 1
        assert out_csv.exists()
        is_valid, errors = validate_submission_file(out_csv, output_format="inchikey14")
        assert is_valid, f"Validation failed: {errors}"


# ===========================================================================
# 3. Fragment Library Injection & Track K Assembly Tests
# ===========================================================================


class TestFragmentLibraryInjection:
    """Test fragment_library injection and Track K candidate aggregation."""

    def test_fragment_library_injection_aggregates_track_knapsack(
        self, synthetic_fragment_library: Dict[int, List[str]], tmp_path: Path
    ):
        test_df = pd.DataFrame({
            "id": ["TEST_KNAP_001"],
            "precursor_mz": [301.071],
            "adduct": ["[M+H]+"],
            "polarity": ["positive"],
        })
        out_csv = tmp_path / "sub_knap_test.csv"

        with CASMIOmegaPipeline(fragment_library=synthetic_fragment_library) as pipeline:
            res_df = pipeline.run(test_df, out_csv, output_format="inchikey14")

        assert len(res_df) == 1
        assert hasattr(pipeline, "last_aggregated_cands")
        knapsack_cands = [
            c for c in pipeline.last_aggregated_cands
            if c.get("source") == "track_knapsack"
        ]
        assert len(knapsack_cands) > 0, "Expected at least one Track K candidate in aggregated_cands"

        for cand in knapsack_cands:
            assert cand["source"] == "track_knapsack"
            assert cand["score"] == 0.3
            ik14 = cand["inchikey14"]
            assert len(ik14) == 14
            assert ik14.isalnum()

    def test_knapsack_fallback_hash_for_string_identifiers(self, tmp_path: Path):
        """Verify non-SMILES string fragment identifiers receive valid 14-char alphanumeric InChIKeys."""
        target_mass = 270.0527235
        target_key = int(round(target_mass * 10000))
        custom_lib = {target_key: ["CUSTOM_FRAG_STRING_ABC"]}

        test_df = pd.DataFrame({
            "id": ["TEST_CUSTOM_001"],
            "precursor_mz": [271.060],
            "adduct": ["[M+H]+"],
            "polarity": ["positive"],
        })
        out_csv = tmp_path / "sub_custom_knap.csv"

        with CASMIOmegaPipeline(fragment_library=custom_lib) as pipeline:
            res_df = pipeline.run(test_df, out_csv, output_format="inchikey14")

        knapsack_cands = [
            c for c in pipeline.last_aggregated_cands
            if c.get("source") == "track_knapsack"
        ]
        assert len(knapsack_cands) > 0
        cand = knapsack_cands[0]
        assert cand["inchikey14"].startswith("IK14KNAP") or len(cand["inchikey14"]) == 14
        assert len(cand["inchikey14"]) == 14
        assert cand["inchikey14"].isalnum()


# ===========================================================================
# 4. 5-Spectrum End-to-End Dry-Run (< 15s) and Output Validation
# ===========================================================================


class TestEndToEndDryRunPhase3:
    """Test 5-spectrum end-to-end execution speed and strict submission schema invariants."""

    def test_5_spectrum_e2e_under_15s(
        self,
        sample_5_spectra_df: pd.DataFrame,
        synthetic_fragment_library: Dict[int, List[str]],
        tmp_path: Path,
    ):
        out_csv = tmp_path / "submission_5spec.csv"

        pipeline = CASMIOmegaPipeline(fragment_library=synthetic_fragment_library)

        start_time = time.perf_counter()
        with pipeline:
            res_df = pipeline.run(sample_5_spectra_df, out_csv, output_format="inchikey14")
        elapsed = time.perf_counter() - start_time

        assert elapsed < 15.0, f"5-spectrum E2E dry run took {elapsed:.2f}s, expected < 15.0s"
        assert len(res_df) == 5

    def test_e2e_output_validation_inchikey14(
        self,
        sample_5_spectra_df: pd.DataFrame,
        synthetic_fragment_library: Dict[int, List[str]],
        tmp_path: Path,
    ):
        out_csv = tmp_path / "submission_ik14.csv"

        with CASMIOmegaPipeline(fragment_library=synthetic_fragment_library) as pipeline:
            pipeline.run(sample_5_spectra_df, out_csv, output_format="inchikey14")

        assert out_csv.exists()
        is_valid, errors = validate_submission_file(
            out_csv,
            expected_ids=sample_5_spectra_df["id"].tolist(),
            output_format="inchikey14",
        )
        assert is_valid, f"Submission validation failed: {errors}"

        # Direct file inspection
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert lines[0] == "id,candidates"
        assert len(lines) == 6  # header + 5 rows

        for row_line in lines[1:]:
            parts = row_line.split(",")
            assert len(parts) == 2
            cand_slots = parts[1].split(";")
            assert len(cand_slots) == 25
            assert len(set(cand_slots)) == 25, "Output slots must be pairwise distinct"
            for slot in cand_slots:
                assert len(slot) == 14
                assert slot.isalnum()

    def test_e2e_output_validation_smiles(
        self,
        sample_5_spectra_df: pd.DataFrame,
        synthetic_fragment_library: Dict[int, List[str]],
        tmp_path: Path,
    ):
        out_csv = tmp_path / "submission_smiles.csv"

        with CASMIOmegaPipeline(fragment_library=synthetic_fragment_library) as pipeline:
            pipeline.run(sample_5_spectra_df, out_csv, output_format="smiles")

        assert out_csv.exists()
        is_valid, errors = validate_submission_file(
            out_csv,
            expected_ids=sample_5_spectra_df["id"].tolist(),
            output_format="smiles",
        )
        assert is_valid, f"SMILES submission validation failed: {errors}"

        # Direct file inspection
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert lines[0] == "molecule_id,smiles"
        assert len(lines) == 6

        # Check validity with RDKit if available
        has_rdkit = False
        try:
            from rdkit import Chem
            has_rdkit = True
        except ImportError:
            pass

        for row_line in lines[1:]:
            parts = row_line.split(",")
            assert len(parts) == 2
            smiles_slots = parts[1].split(";")
            assert len(smiles_slots) == 25
            assert len(set(smiles_slots)) == 25

            if has_rdkit:
                for smi in smiles_slots:
                    mol = Chem.MolFromSmiles(smi)
                    assert mol is not None, f"Expected valid SMILES, got invalid string '{smi}'"


# ===========================================================================
# 5. Runner Wrapper and Brand Aliases
# ===========================================================================


class TestPipelineRunnerWrappers:
    """Test run_casmi_omega_pipeline and run_phytoforge_pipeline wrapper integration."""

    def test_run_casmi_omega_pipeline_passes_all_kwargs(
        self,
        sample_5_spectra_df: pd.DataFrame,
        synthetic_fragment_library: Dict[int, List[str]],
        minimal_sqlite_db: Path,
        tmp_path: Path,
    ):
        out_csv = tmp_path / "wrapper_out.csv"

        res_df = run_casmi_omega_pipeline(
            test_df=sample_5_spectra_df.iloc[:2],
            output_path=out_csv,
            db_path=minimal_sqlite_db,
            fragment_library=synthetic_fragment_library,
            output_format="inchikey14",
        )

        assert len(res_df) == 2
        assert out_csv.exists()
        is_valid, errors = validate_submission_file(out_csv, output_format="inchikey14")
        assert is_valid, f"Validation failed: {errors}"

    def test_run_phytoforge_pipeline_alias(
        self,
        sample_5_spectra_df: pd.DataFrame,
        tmp_path: Path,
    ):
        out_csv = tmp_path / "alias_out.csv"
        res_df = run_phytoforge_pipeline(
            test_df=sample_5_spectra_df.iloc[:1],
            output_path=out_csv,
            output_format="smiles",
        )
        assert len(res_df) == 1
        assert out_csv.exists()
        is_valid, errors = validate_submission_file(out_csv, output_format="smiles")
        assert is_valid, f"Validation failed: {errors}"
