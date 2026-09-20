"""
tests/test_indexer_cli.py
Comprehensive unit and integration test suite for the unified asset indexer CLI
and validation engine (Milestone M5, R4).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data.indexer import (
    build_parser,
    main,
    run_tanimoto_benchmark,
    validate_chemical_database,
    validate_indices,
    validate_spectral_index,
)


# ---------------------------------------------------------------------------
# Helpers for Synthetic CLI Input Files
# ---------------------------------------------------------------------------

def write_sample_molecules_csv(path: Path) -> Path:
    """Writes a sample molecules CSV for testing index-db."""
    molecules = [
        ("MOL_001", "CC(=O)Oc1ccccc1C(=O)O"),  # Aspirin
        ("MOL_002", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),  # Caffeine
        ("MOL_003", "c1ccccc1O"),  # Phenol
        ("MOL_004", "c1ccccc1"),  # Benzene
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "smiles"])
        for mid, smi in molecules:
            writer.writerow([mid, smi])
    return path


def write_sample_spectra_parquet(path: Path, dim: int = 1024) -> Path:
    """Writes a sample spectra Parquet file for testing index-spectra."""
    rng = np.random.default_rng(42)
    records = []
    mzs = [150.05, 200.10, 250.15, 300.20, 180.08, 220.12]
    adducts = ["[M+H]+", "[M+H]+", "[M+Na]+", "[M+H]+", "[M-H]-", "[M-H]-"]
    polarities = ["positive", "positive", "positive", "positive", "negative", "negative"]

    for i in range(len(mzs)):
        v = rng.standard_normal(dim).astype(np.float32)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        records.append(
            {
                "id": f"SPEC_{i:04d}",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "inchikey14": "BSYNRYMUTXBXSQ",
                "precursor_mz": mzs[i],
                "adduct": adducts[i],
                "instrument": "timsTOF",
                "polarity": polarities[i],
                "embedding": list(v),
            }
        )
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return path


# ===========================================================================
# 1. CLI Help and Subcommand Parsing Tests
# ===========================================================================

class TestCLIHelpAndParsers:
    """Test CLI argument parsing, --help displays, and subcommand registration."""

    def test_cli_root_help(self):
        cmd = [sys.executable, "-m", "src.data.indexer", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0
        assert "PhytoForge offline asset indexing" in res.stdout
        assert "index-db" in res.stdout
        assert "index-spectra" in res.stdout
        assert "validate-index" in res.stdout
        assert "benchmark" in res.stdout

    @pytest.mark.parametrize("subcmd", ["index-db", "index-spectra", "validate-index", "benchmark"])
    def test_cli_subcommand_help(self, subcmd: str):
        cmd = [sys.executable, "-m", "src.data.indexer", subcmd, "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0
        assert f"usage: python -m src.data.indexer {subcmd}" in res.stdout

    def test_cli_no_subcommand_prints_help_and_exits_1(self):
        cmd = [sys.executable, "-m", "src.data.indexer"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        assert "usage: python -m src.data.indexer" in res.stderr

    def test_cli_unknown_subcommand_exits_2(self):
        cmd = [sys.executable, "-m", "src.data.indexer", "nonexistent-subcommand"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 2


# ===========================================================================
# 2. End-to-End index-db Subcommand Tests
# ===========================================================================

class TestIndexDbSubcommand:
    """Test end-to-end execution of index-db subcommand."""

    def test_index_db_from_csv_success(self, tmp_path: Path):
        csv_file = write_sample_molecules_csv(tmp_path / "sample_mols.csv")
        out_db = tmp_path / "output.db"
        dead_letter = tmp_path / "dead_letter.jsonl"

        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "index-db",
            "--input",
            str(csv_file),
            "--output",
            str(out_db),
            "--dead-letter-log",
            str(dead_letter),
            "--batch-size",
            "2",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"index-db failed with stderr: {res.stderr}"
        assert "Successfully indexed 4 molecules into" in res.stdout
        assert out_db.exists()

        # Validate database directly
        is_valid, msgs = validate_chemical_database(out_db)
        assert is_valid is True, f"Validation failed: {msgs}"

    def test_index_db_from_smiles_list_file(self, tmp_path: Path):
        smi_file = tmp_path / "smiles.txt"
        with open(smi_file, "w", encoding="utf-8") as f:
            f.write("CC(=O)Oc1ccccc1C(=O)O\n")
            f.write("c1ccccc1O\n")
            f.write("c1ccccc1\n")

        out_db = tmp_path / "smiles_out.db"
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "index-db",
            "--input",
            str(smi_file),
            "--output",
            str(out_db),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0
        assert "Successfully indexed 3 molecules" in res.stdout

    def test_index_db_missing_input_file_fails(self, tmp_path: Path):
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "index-db",
            "--input",
            str(tmp_path / "nonexistent.csv"),
            "--output",
            str(tmp_path / "fail.db"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        assert "Input file does not exist" in res.stderr


# ===========================================================================
# 3. End-to-End index-spectra Subcommand Tests
# ===========================================================================

class TestIndexSpectraSubcommand:
    """Test end-to-end execution of index-spectra subcommand."""

    def test_index_spectra_success(self, tmp_path: Path):
        pq_file = write_sample_spectra_parquet(tmp_path / "spectra.parquet", dim=1024)
        out_dir = tmp_path / "spectral_index_out"

        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "index-spectra",
            "--input",
            str(pq_file),
            "--output-dir",
            str(out_dir),
            "--dim",
            "1024",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"index-spectra failed with stderr: {res.stderr}"
        assert "Successfully indexed spectra into" in res.stdout
        assert (out_dir / "manifest.json").exists()

        # Validate spectral index
        is_valid, msgs = validate_spectral_index(out_dir)
        assert is_valid is True, f"Spectral index validation failed: {msgs}"

    def test_index_spectra_missing_input_file_fails(self, tmp_path: Path):
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "index-spectra",
            "--input",
            str(tmp_path / "nonexistent.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        assert "Input file does not exist" in res.stderr


# ===========================================================================
# 4. validate-index Subcommand & Unit Validation Tests
# ===========================================================================

class TestValidateIndexSubcommand:
    """Test index validation CLI subcommand and programmatic validation logic."""

    @pytest.fixture
    def generated_db_and_spectral(self, tmp_path: Path):
        # Generate DB
        csv_file = write_sample_molecules_csv(tmp_path / "mols.csv")
        out_db = tmp_path / "valid.db"
        subprocess.run(
            [sys.executable, "-m", "src.data.indexer", "index-db", "--input", str(csv_file), "--output", str(out_db)],
            check=True,
        )

        # Generate Spectral Index
        pq_file = write_sample_spectra_parquet(tmp_path / "spectra.parquet")
        out_sp = tmp_path / "spectral_index"
        subprocess.run(
            [sys.executable, "-m", "src.data.indexer", "index-spectra", "--input", str(pq_file), "--output-dir", str(out_sp)],
            check=True,
        )

        return out_db, out_sp

    def test_cli_validate_db_success(self, generated_db_and_spectral):
        out_db, _ = generated_db_and_spectral
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "validate-index",
            "--db-path",
            str(out_db),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0
        assert "Index validation successful" in res.stdout

    def test_cli_validate_spectral_success(self, generated_db_and_spectral):
        _, out_sp = generated_db_and_spectral
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "validate-index",
            "--spectral-dir",
            str(out_sp),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0
        assert "Index validation successful" in res.stdout

    def test_cli_validate_both_success(self, generated_db_and_spectral):
        out_db, out_sp = generated_db_and_spectral
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "validate-index",
            "--db-path",
            str(out_db),
            "--spectral-dir",
            str(out_sp),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0
        assert "Index validation successful" in res.stdout

    def test_cli_validate_nonexistent_files_fails(self, tmp_path: Path):
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "validate-index",
            "--db-path",
            str(tmp_path / "nonexistent.db"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        assert "Index validation FAILED" in res.stderr

    def test_validate_chemical_database_detects_corrupted_schema(self, tmp_path: Path):
        """Verify validation catches tables without WITHOUT ROWID or missing columns."""
        corrupt_db = tmp_path / "corrupt.db"
        conn = sqlite3.connect(str(corrupt_db))
        # Create normal table WITH rowid (not WITHOUT ROWID)
        conn.execute("CREATE TABLE candidates (formula TEXT, inchikey14 TEXT, smiles TEXT, fingerprint BLOB);")
        conn.commit()
        conn.close()

        is_valid, errors = validate_chemical_database(corrupt_db)
        assert is_valid is False
        assert any("NOT created WITHOUT ROWID" in err for err in errors)
        assert any("Table '_schema_metadata' not found" in err for err in errors)

    def test_validate_spectral_index_detects_missing_manifest(self, tmp_path: Path):
        bad_dir = tmp_path / "empty_dir"
        bad_dir.mkdir()
        is_valid, errors = validate_spectral_index(bad_dir)
        assert is_valid is False
        assert any("manifest.json not found" in err for err in errors)


# ===========================================================================
# 5. Benchmark Subcommand Tests
# ===========================================================================

class TestBenchmarkSubcommand:
    """Test popcount Tanimoto scoring benchmark execution and latency check."""

    def test_cli_benchmark_default(self):
        cmd = [
            sys.executable,
            "-m",
            "src.data.indexer",
            "benchmark",
            "--num-candidates",
            "10000",
            "--iterations",
            "20",
            "--threshold-ms",
            "2.0",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"Benchmark failed: {res.stderr}"
        assert "Benchmark result:" in res.stdout
        assert "Benchmark PASSED." in res.stdout

    def test_run_tanimoto_benchmark_programmatic(self):
        passed, latency = run_tanimoto_benchmark(num_candidates=5000, iterations=10, threshold_ms=2.0)
        assert passed is True
        assert latency < 2.0
        assert latency > 0.0
