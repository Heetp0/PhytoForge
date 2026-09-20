"""
tests/test_adversarial_challenger_2.py
Adversarial challenge test suite for CASMIOmegaPipeline (Phase 3 Challenger 2):
1. Context manager re-entrancy and double close() idempotency (SQLite connections & mmap handles).
2. 5-spectrum E2E dry-run with dual format submission ("inchikey14" and "smiles") validated by validator CLI.
3. Resilience against corrupted spectra (empty peaks, negative intensities, missing/invalid precursor m/z).
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.pipeline import CASMIOmegaPipeline, run_casmi_omega_pipeline
from src.reranking.fragment_library import KnapsackAssembler, NeutralLossLibrary, parse_peaks_field
from src.submission.writer import validate_submission, validate_submission_file
from src.data.spectral_indexer import SpectralIndexBuilder


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_disk_db(tmp_path: Path) -> Path:
    """Creates a real SQLite file database on disk."""
    db_file = tmp_path / "compounds_test.db"
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
    fp_blob = np.zeros(32, dtype=np.uint64).tobytes()
    conn.execute(
        "INSERT INTO candidates VALUES (?, ?, ?, ?)",
        ("C15H10O5", "IK14DBFLAV0001", "c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1", fp_blob),
    )
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def mock_spectral_index(tmp_path: Path) -> Path:
    """Creates a real partitioned SpectralIndex with mmap embeddings and metadata on disk."""
    spec_dir = tmp_path / "spectral_index"
    embs = np.random.randn(3, 1024).astype(np.float32)
    meta = pd.DataFrame({
        "id": ["S1", "S2", "S3"],
        "smiles": ["CCO", "CCN", "CCC"],
        "inchikey14": ["IK14SPECA00001", "IK14SPECB00002", "IK14SPECC00003"],
        "precursor_mz": [150.0, 250.0, 350.0],
        "adduct": ["[M+H]+", "[M+H]+", "[M+H]+"],
        "instrument": ["timsTOF", "timsTOF", "timsTOF"],
        "polarity": ["positive", "positive", "positive"],
    })
    SpectralIndexBuilder.build_index(spec_dir, embeddings=embs, metadata=meta)
    return spec_dir


@pytest.fixture
def sample_5_spectra() -> pd.DataFrame:
    """5 realistic test spectra for E2E validation."""
    return pd.DataFrame({
        "id": [f"SPEC_E2E_{i:03d}" for i in range(1, 6)],
        "precursor_mz": [271.060, 287.055, 301.071, 315.086, 447.129],
        "adduct": ["[M+H]+"] * 5,
        "polarity": ["positive"] * 5,
    })


@pytest.fixture
def knapsack_library() -> Dict[int, List[str]]:
    """Synthetic fragment library for Track K assembly."""
    return {
        2700527: ["c1ccccc1O", "c1cc(O)ccc1O"],
        2860477: ["c1cc(O)c(O)cc1O"],
        3000637: ["c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1"],
        3140787: ["COc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1"],
        4461217: ["c1cc(O)c2c(=O)cc(-c3ccc(O)cc3)oc2c1"],
    }


# ===========================================================================
# 1. Context Manager Re-entrancy & Double close() Tests
# ===========================================================================

class TestContextManagerAndDoubleClose:
    """Adversarial stress-testing of context manager and resource disposal."""

    def test_double_close_in_memory(self):
        """Pipeline without disk handles can be closed multiple times without error."""
        pipeline = CASMIOmegaPipeline()
        for _ in range(5):
            pipeline.close()

    def test_double_close_with_disk_db_and_spectral_index(
        self, mock_disk_db: Path, mock_spectral_index: Path
    ):
        """Pipeline with disk SQLite and mmap index is completely idempotent on repeated close()."""
        pipeline = CASMIOmegaPipeline(
            db_path=mock_disk_db,
            spectral_index_path=mock_spectral_index,
        )
        # Verify resources exist before close
        assert pipeline.db_searcher.repo is not None
        assert pipeline.dreams_retriever.index is not None

        # Repeat close 5 times
        for _ in range(5):
            pipeline.close()

        # Check that underlying repository conn is set to None
        assert pipeline.db_searcher.repo.conn is None

    def test_context_manager_nested_reentrancy(
        self, mock_disk_db: Path, mock_spectral_index: Path
    ):
        """Pipeline handles nested with blocks without crashing on exit."""
        pipeline = CASMIOmegaPipeline(
            db_path=mock_disk_db,
            spectral_index_path=mock_spectral_index,
        )
        with pipeline as p1:
            with p1 as p2:
                assert p1 is p2
        # Exit of outer block calls close() again after inner block called close()
        # Should not raise any exception.

    def test_context_manager_sequential_reentrancy(
        self, mock_disk_db: Path, mock_spectral_index: Path
    ):
        """Pipeline handles repeated sequential context manager usage."""
        pipeline = CASMIOmegaPipeline(
            db_path=mock_disk_db,
            spectral_index_path=mock_spectral_index,
        )
        with pipeline:
            pass
        with pipeline:
            pass

    def test_direct_subcomponent_close_then_pipeline_close(
        self, mock_disk_db: Path, mock_spectral_index: Path
    ):
        """If subcomponents are closed externally, pipeline.close() does not crash."""
        pipeline = CASMIOmegaPipeline(
            db_path=mock_disk_db,
            spectral_index_path=mock_spectral_index,
        )
        pipeline.db_searcher.close()
        pipeline.dreams_retriever.close()
        # Now call pipeline.close()
        pipeline.close()
        pipeline.close()


# ===========================================================================
# 2. 5-Spectrum E2E Dry-Run & CLI Validation Tests
# ===========================================================================

class TestFiveSpectrumEndToEndDualFormat:
    """E2E dry-run with dual format submission and strict CLI validation."""

    def test_5_spectrum_e2e_inchikey14_cli_validation(
        self,
        sample_5_spectra: pd.DataFrame,
        knapsack_library: Dict[int, List[str]],
        tmp_path: Path,
    ):
        out_csv = tmp_path / "submission_ik14.csv"
        with CASMIOmegaPipeline(fragment_library=knapsack_library) as p:
            res_df = p.run(sample_5_spectra, output_path=out_csv, output_format="inchikey14")

        assert len(res_df) == 5
        assert out_csv.exists()

        # Validate with CLI tool
        cmd = [
            sys.executable,
            "-m",
            "src.submission.validator",
            "--submission",
            str(out_csv),
            "--format",
            "inchikey14",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Validator CLI failed: {res.stderr}"
        assert "Validation successful" in res.stdout

        # Verify exact 25 unique 14-char alphanumeric slots per row
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert lines[0] == "id,candidates"
        assert len(lines) == 6
        for line in lines[1:]:
            parts = line.split(",")
            assert len(parts) == 2
            slots = parts[1].split(";")
            assert len(slots) == 25
            assert len(set(slots)) == 25
            for s in slots:
                assert len(s) == 14
                assert s.isalnum()

    def test_5_spectrum_e2e_smiles_cli_validation(
        self,
        sample_5_spectra: pd.DataFrame,
        knapsack_library: Dict[int, List[str]],
        tmp_path: Path,
    ):
        out_csv = tmp_path / "submission_smiles.csv"
        with CASMIOmegaPipeline(fragment_library=knapsack_library) as p:
            res_df = p.run(sample_5_spectra, output_path=out_csv, output_format="smiles")

        assert len(res_df) == 5
        assert out_csv.exists()

        # Validate with CLI tool
        cmd = [
            sys.executable,
            "-m",
            "src.submission.validator",
            "--submission",
            str(out_csv),
            "--format",
            "smiles",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Validator CLI failed: {res.stderr}"
        assert "Validation successful" in res.stdout

        # Verify exact 25 unique slots per row
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert lines[0] == "molecule_id,smiles"
        assert len(lines) == 6
        for line in lines[1:]:
            parts = line.split(",")
            assert len(parts) == 2
            slots = parts[1].split(";")
            assert len(slots) == 25
            assert len(set(slots)) == 25
            for s in slots:
                assert s.strip() != ""
                assert not any(c in s for c in (" ", "\t", "\n", "\r", ","))


# ===========================================================================
# 3. Resilience Against Corrupted / Pathological Spectra
# ===========================================================================

class TestCorruptedSpectraResilience:
    """Adversarial stress-testing on malformed, missing, or negative inputs."""

    @pytest.fixture
    def pathological_spectra_df(self) -> pd.DataFrame:
        """Synthetic DataFrame containing diverse pathological edge cases."""
        return pd.DataFrame([
            {
                "id": "PATH_EMPTY_PEAKS",
                "precursor_mz": 300.0,
                "adduct": "[M+H]+",
                "polarity": "positive",
                "peaks": "",
            },
            {
                "id": "PATH_NEG_INTENSITY",
                "precursor_mz": 280.0,
                "adduct": "[M+H]+",
                "polarity": "positive",
                "peaks": "100.0:-10.0;150.0:-50.0",
            },
            {
                "id": "PATH_MISSING_MZ_NAN",
                "precursor_mz": np.nan,
                "adduct": "[M+H]+",
                "polarity": "positive",
                "peaks": "120.0:40.0;140.0:80.0",
            },
            {
                "id": "PATH_MISSING_MZ_NONE",
                "precursor_mz": None,
                "adduct": "[M-H]-",
                "polarity": "negative",
                "peaks": None,
            },
            {
                "id": "PATH_NEGATIVE_MZ",
                "precursor_mz": -500.0,
                "adduct": "[M+H]+",
                "polarity": "positive",
                "peaks": np.empty((0, 2)),
            },
            {
                "id": "PATH_ZERO_MZ",
                "precursor_mz": 0.0,
                "adduct": "UNKNOWN_ADDUCT",
                "polarity": "unknown",
                "peaks": ";;;;",
            },
            {
                "id": "PATH_STRING_MZ",
                "precursor_mz": "NOT_NUMERIC",
                "adduct": "",
                "polarity": "",
                "peaks": [],
            },
        ])

    def test_pipeline_runs_to_completion_on_pathological_spectra(
        self, pathological_spectra_df: pd.DataFrame, tmp_path: Path
    ):
        """Pipeline must gracefully handle missing/corrupt values and produce valid submission CSV."""
        out_ik = tmp_path / "pathological_ik14.csv"
        out_smi = tmp_path / "pathological_smiles.csv"

        with CASMIOmegaPipeline() as p:
            df_ik = p.run(pathological_spectra_df, output_path=out_ik, output_format="inchikey14")
            df_smi = p.run(pathological_spectra_df, output_path=out_smi, output_format="smiles")

        assert len(df_ik) == len(pathological_spectra_df)
        assert len(df_smi) == len(pathological_spectra_df)

        is_valid_ik, errors_ik = validate_submission_file(out_ik, output_format="inchikey14")
        assert is_valid_ik, f"InChIKey14 validation failed on pathological spectra: {errors_ik}"

        is_valid_smi, errors_smi = validate_submission_file(out_smi, output_format="smiles")
        assert is_valid_smi, f"SMILES validation failed on pathological spectra: {errors_smi}"

    def test_neutral_loss_library_discards_pathological_spectra(
        self, pathological_spectra_df: pd.DataFrame
    ):
        """NeutralLossLibrary silently ignores corrupt/negative spectra without crashing."""
        library = NeutralLossLibrary()
        stats = library.build_from_dataframe(pathological_spectra_df)
        assert stats["total_spectra"] == len(pathological_spectra_df)
        assert stats["total_fragments"] == 0
        assert len(library) == 0

    def test_parse_peaks_field_edge_cases(self):
        """parse_peaks_field returns empty list on invalid, unparseable, or empty peak inputs."""
        assert parse_peaks_field(None) == []
        assert parse_peaks_field("") == []
        assert parse_peaks_field("   ") == []
        assert parse_peaks_field(";;;;") == []
        assert parse_peaks_field(np.empty((0, 2))) == []
        assert parse_peaks_field([[], []]) == []

        # Negative intensities are parsed faithfully without crashing
        parsed = parse_peaks_field("100.0:-10.0;150.0:-50.0")
        assert parsed == [(100.0, -10.0), (150.0, -50.0)]
