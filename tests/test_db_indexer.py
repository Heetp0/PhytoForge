"""
tests/test_db_indexer.py
Comprehensive unit and integration test suite for SQLite-backed chemical database
indexer and repository (Milestone M5, R1 & R4 DI).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.chemistry.standardizer import (
    compute_hill_formula,
    compute_morgan_fingerprint,
    standardize_mol,
)
from src.data.db_indexer import (
    ChemicalDatabaseIndexer,
    ChemicalDatabaseRepository,
    pack_fingerprint,
    popcount_tanimoto,
    unpack_fingerprint,
)
from src.retrieval.database_search import DBCandidate, SoftDatabaseSearcher


# ---------------------------------------------------------------------------
# Synthetic Molecule Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_MOLECULES = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C8H10N4O2"),
    ("Benzoic_Acid", "c1ccccc1C(=O)O", "C7H6O2"),
    ("Phenol", "c1ccccc1O", "C6H6O"),
    ("Toluene", "Cc1ccccc1", "C7H8"),
    ("Benzene", "c1ccccc1", "C6H6"),
    ("Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O", "C6H12O6"),
    ("Resveratrol", "Oc1cc(O)cc(/C=C/c2ccc(O)cc2)c1", "C14H12O3"),
    ("Quercetin", "c1cc(c(cc1O)O)C2=C(C(=O)c3c(cc(cc3O2)O)O)O", "C15H10O7"),
    ("Paracetamol", "CC(=O)Nc1ccc(cc1)O", "C8H9NO2"),
]


# ===========================================================================
# 1. Fingerprint Packing and Unpacking Tests
# ===========================================================================

class TestFingerprintPacking:
    """Test 2048-bit Morgan fingerprint packing into 32 uint64 words and unpacking."""

    def test_pack_fingerprint_1d_shape_and_dtype(self):
        arr = np.zeros(2048, dtype=np.uint8)
        arr[0] = 1
        arr[63] = 1
        arr[64] = 1
        arr[2047] = 1

        packed = pack_fingerprint(arr)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (32,)
        assert packed.dtype == np.uint64

        # Verify bit 0 is set in word 0
        assert (packed[0] & np.uint64(1)) != 0
        # Verify bit 63 is set in word 0
        assert (packed[0] & (np.uint64(1) << np.uint64(63))) != 0
        # Verify bit 0 in word 1 corresponds to index 64
        assert (packed[1] & np.uint64(1)) != 0
        # Verify bit 63 in word 31 corresponds to index 2047
        assert (packed[31] & (np.uint64(1) << np.uint64(63))) != 0

    def test_pack_fingerprint_2d_shape_and_dtype(self):
        mat = np.zeros((15, 2048), dtype=bool)
        mat[0, 5] = True
        mat[3, 100] = True
        mat[14, 2040] = True

        packed = pack_fingerprint(mat)
        assert packed.shape == (15, 32)
        assert packed.dtype == np.uint64

    def test_roundtrip_pack_unpack_1d(self):
        rng = np.random.default_rng(42)
        original = rng.choice([0, 1], size=2048, p=[0.9, 0.1]).astype(np.uint8)

        packed = pack_fingerprint(original)
        unpacked = unpack_fingerprint(packed)

        assert unpacked.shape == (2048,)
        assert unpacked.dtype == np.uint8
        np.testing.assert_array_equal(original, unpacked)

    def test_roundtrip_pack_unpack_2d(self):
        rng = np.random.default_rng(123)
        original = rng.choice([0, 1], size=(20, 2048), p=[0.85, 0.15]).astype(np.uint8)

        packed = pack_fingerprint(original)
        unpacked = unpack_fingerprint(packed)

        assert unpacked.shape == (20, 2048)
        np.testing.assert_array_equal(original, unpacked)

    def test_pack_already_packed_returns_copy(self):
        already_packed = np.ones(32, dtype=np.uint64)
        res = pack_fingerprint(already_packed)
        assert res.shape == (32,)
        assert res.dtype == np.uint64
        assert res is not already_packed
        np.testing.assert_array_equal(already_packed, res)

        already_packed_2d = np.ones((5, 32), dtype=np.uint64)
        res_2d = pack_fingerprint(already_packed_2d)
        assert res_2d.shape == (5, 32)
        assert res_2d is not already_packed_2d

    def test_pack_invalid_dimensions_raises_error(self):
        with pytest.raises(ValueError, match="Expected 2048 elements"):
            pack_fingerprint(np.zeros(1024, dtype=np.uint8))

        with pytest.raises(ValueError, match="Expected shape .*2048"):
            pack_fingerprint(np.zeros((5, 512), dtype=np.uint8))

        with pytest.raises(ValueError, match="Unsupported array dimension"):
            pack_fingerprint(np.zeros((2, 2, 2048), dtype=np.uint8))

    def test_unpack_invalid_dimensions_raises_error(self):
        with pytest.raises(ValueError, match="Expected 32 words"):
            unpack_fingerprint(np.zeros(16, dtype=np.uint64))

        with pytest.raises(ValueError, match="Expected shape .*32"):
            unpack_fingerprint(np.zeros((5, 16), dtype=np.uint64))

        with pytest.raises(ValueError, match="Unsupported packed array dimension"):
            unpack_fingerprint(np.zeros((2, 2, 32), dtype=np.uint64))


# ===========================================================================
# 2. Popcount Tanimoto Scoring Tests
# ===========================================================================

class TestPopcountTanimoto:
    """Test vectorized popcount Tanimoto similarity scoring."""

    def test_exact_match_yields_similarity_one(self):
        rng = np.random.default_rng(999)
        raw_fp = rng.choice([0, 1], size=2048, p=[0.9, 0.1]).astype(np.uint8)
        packed_q = pack_fingerprint(raw_fp)
        candidates = np.stack([packed_q, packed_q, packed_q], axis=0)

        scores = popcount_tanimoto(packed_q, candidates)
        assert scores.shape == (3,)
        np.testing.assert_allclose(scores, [1.0, 1.0, 1.0], atol=1e-5)

    def test_disjoint_fingerprints_yield_similarity_zero(self):
        # Query has even bits set, candidate has odd bits set
        fp_q = np.zeros(2048, dtype=np.uint8)
        fp_q[::2] = 1
        fp_cand = np.zeros(2048, dtype=np.uint8)
        fp_cand[1::2] = 1

        packed_q = pack_fingerprint(fp_q)
        packed_cand = pack_fingerprint(fp_cand)
        candidates = packed_cand.reshape(1, 32)

        scores = popcount_tanimoto(packed_q, candidates)
        assert len(scores) == 1
        assert scores[0] == 0.0

    def test_analytical_tanimoto_overlap(self):
        # Construct known overlap:
        # Query has 6 bits set at indices 0, 1, 2, 3, 4, 5
        # Cand 1 has 6 bits set at indices 3, 4, 5, 6, 7, 8 -> intersection 3, union 9 -> 3/9 = 0.33333
        # Cand 2 has 4 bits set at indices 0, 1, 2, 3 -> intersection 4, union 6 -> 4/6 = 0.66667
        fp_q = np.zeros(2048, dtype=np.uint8)
        fp_q[0:6] = 1

        fp_c1 = np.zeros(2048, dtype=np.uint8)
        fp_c1[3:9] = 1

        fp_c2 = np.zeros(2048, dtype=np.uint8)
        fp_c2[0:4] = 1

        packed_q = pack_fingerprint(fp_q)
        cands = np.stack([pack_fingerprint(fp_c1), pack_fingerprint(fp_c2)], axis=0)

        scores = popcount_tanimoto(packed_q, cands)
        np.testing.assert_allclose(scores, [3.0 / 9.0, 4.0 / 6.0], atol=1e-5)

    def test_unpacked_query_vector_auto_packed(self):
        fp_q = np.zeros(2048, dtype=np.uint8)
        fp_q[10] = 1
        fp_cand = np.zeros(2048, dtype=np.uint8)
        fp_cand[10] = 1
        packed_cands = pack_fingerprint(fp_cand).reshape(1, 32)

        # Pass 2048-dim unpacked array as query
        scores = popcount_tanimoto(fp_q, packed_cands)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_empty_candidates_returns_empty_array(self):
        q = np.zeros(32, dtype=np.uint64)
        assert len(popcount_tanimoto(q, np.empty((0, 32), dtype=np.uint64))) == 0
        assert len(popcount_tanimoto(q, None)) == 0

    def test_1d_candidate_reshaped_automatically(self):
        fp = np.zeros(32, dtype=np.uint64)
        fp[0] = np.uint64(5)
        scores = popcount_tanimoto(fp, fp)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_benchmark_10k_candidates_sub_2ms(self):
        """Verify 10,000 candidate scoring completes under 2.0 ms."""
        rng = np.random.default_rng(2026)
        # Generate random 64-bit integers for 10k candidates
        high_63 = (1 << 63) - 1
        candidate_fps = rng.integers(0, high_63, size=(10000, 32), dtype=np.uint64)
        query_fp = rng.integers(0, high_63, size=32, dtype=np.uint64)

        # Warmup (multiple iterations to ensure Numba JIT and thread pool are initialized)
        for _ in range(3):
            _ = popcount_tanimoto(query_fp, candidate_fps)

        # Measure 20 iterations
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            scores = popcount_tanimoto(query_fp, candidate_fps)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_latency_ms = float(np.mean(times))
        assert len(scores) == 10000
        assert all(0.0 <= s <= 1.0 for s in scores[:100])
        # Kaggle requirement: < 2.0 ms
        assert mean_latency_ms < 2.0, f"Mean latency was {mean_latency_ms:.3f} ms, expected < 2.0 ms"


# ===========================================================================
# 3. Canonical Hill Formula Calculation Tests
# ===========================================================================

class TestHillFormulaCalculation:
    """Test canonical Hill system formula generation."""

    @pytest.mark.parametrize(
        "name,smiles,expected_formula",
        SYNTHETIC_MOLECULES,
    )
    def test_hill_formula_known_compounds(self, name, smiles, expected_formula):
        formula = compute_hill_formula(smiles)
        assert formula == expected_formula, f"Failed for {name}: expected {expected_formula}, got {formula}"

    def test_hill_formula_without_carbon(self):
        # Compounds without Carbon are ordered purely alphabetically
        water_formula = compute_hill_formula("O")
        assert water_formula == "H2O"

        hcl_formula = compute_hill_formula("Cl")
        assert hcl_formula == "HCl"  # H before Cl alphabetically

    def test_hill_formula_invalid_inputs(self):
        assert compute_hill_formula(None) is None
        assert compute_hill_formula("") is None
        assert compute_hill_formula("   ") is None
        assert compute_hill_formula("INVALID_SYNTAX_99") is None


# ===========================================================================
# 4. ChemicalDatabaseIndexer Tests
# ===========================================================================

class TestChemicalDatabaseIndexer:
    """Test SQLite database indexer schema, pragmas, and ingestion."""

    def test_schema_and_pragmas(self, tmp_path: Path):
        db_path = tmp_path / "test_schema.db"
        indexer = ChemicalDatabaseIndexer(db_path=db_path)

        # Verify tables exist and are WITHOUT ROWID
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='candidates';")
        sql_cand = cursor.fetchone()[0]
        assert "WITHOUT ROWID" in sql_cand.upper()
        assert "PRIMARY KEY (formula, inchikey14)" in sql_cand

        cursor.execute("PRAGMA table_info(candidates);")
        cols = {row[1]: row[2] for row in cursor.fetchall()}
        assert cols == {
            "formula": "TEXT",
            "inchikey14": "TEXT",
            "smiles": "TEXT",
            "fingerprint": "BLOB",
        }

        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='_schema_metadata';")
        sql_meta = cursor.fetchone()[0]
        assert "WITHOUT ROWID" in sql_meta.upper()

        # Verify initial metadata
        meta = indexer.get_metadata()
        assert meta.get("schema_version") == "1.0"
        assert meta.get("fingerprint_type") == "Morgan_r2_b2048_uint64x32"
        assert "created_at" in meta

        conn.close()
        indexer.close()

    def test_index_from_smiles_list(self, tmp_path: Path):
        db_path = tmp_path / "test_smiles_list.db"
        smiles_list = [item[1] for item in SYNTHETIC_MOLECULES]

        with ChemicalDatabaseIndexer(db_path=db_path) as indexer:
            stats = indexer.index_from_smiles_list(smiles_list, batch_size=5)

        assert stats["indexed"] == len(smiles_list)
        assert stats["failed"] == 0
        assert stats["total_in_db"] == len(smiles_list)

        # Inspect database content
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT formula, inchikey14, smiles, length(fingerprint) FROM candidates;")
        rows = cursor.fetchall()
        assert len(rows) == len(smiles_list)
        for formula, ik14, smiles, fp_len in rows:
            assert len(formula) > 0
            assert len(ik14) == 14
            assert len(smiles) > 0
            assert fp_len == 256  # 32 uint64 words = 256 bytes
        conn.close()

    def test_index_from_csv(self, tmp_path: Path):
        csv_path = tmp_path / "molecules.csv"
        db_path = tmp_path / "test_from_csv.db"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "smiles", "name"])
            for name, smiles, _ in SYNTHETIC_MOLECULES:
                writer.writerow([f"MOL_{name}", smiles, name])

        indexer = ChemicalDatabaseIndexer(db_path=db_path)
        stats = indexer.index_from_csv(csv_path, smiles_col="smiles", id_col="id", batch_size=4)
        indexer.close()

        assert stats["indexed"] == len(SYNTHETIC_MOLECULES)
        assert stats["total_in_db"] == len(SYNTHETIC_MOLECULES)

    def test_dead_letter_quarantine_resilience(self, tmp_path: Path):
        """Verify unparseable SMILES are safely quarantined without failing batch."""
        db_path = tmp_path / "test_dead_letters.db"
        dead_letter_file = tmp_path / "dead_letters.jsonl"

        records = [
            {"id": "V1", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},  # Valid Aspirin
            {"id": "INV1", "smiles": "NOT_A_VALID_SMILES_123"},  # Invalid
            {"id": "V2", "smiles": "c1ccccc1O"},  # Valid Phenol
            {"id": "EMPTY", "smiles": ""},  # Empty string
            {"id": "NONE", "smiles": None},  # None
            {"id": "V3", "smiles": "c1ccccc1"},  # Valid Benzene
        ]

        indexer = ChemicalDatabaseIndexer(
            db_path=db_path,
            dead_letter_path=dead_letter_file,
        )
        stats = indexer.index_molecules(records, batch_size=2)
        indexer.close()

        assert stats["indexed"] == 3
        assert stats["failed"] == 3
        assert stats["total"] == 6
        assert len(indexer.dead_letters) == 3

        # Verify dead letter JSONL contents
        assert dead_letter_file.exists()
        dead_entries = []
        with open(dead_letter_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dead_entries.append(json.loads(line))

        assert len(dead_entries) == 3
        reasons = [e["reason"] for e in dead_entries]
        assert any("Standardization failed" in r for r in reasons)
        assert any("Empty or invalid SMILES" in r for r in reasons)

    def test_suppress_chiral_flag_deduplication(self, tmp_path: Path):
        """Verify stereoisomers deduplicate to same 2D connectivity with suppress_chiral=True."""
        db_path = tmp_path / "test_chiral.db"
        # L-lactic acid and D-lactic acid
        records = [
            {"id": "L-lactate", "smiles": "C[C@@H](O)C(=O)O"},
            {"id": "D-lactate", "smiles": "C[C@H](O)C(=O)O"},
        ]

        indexer = ChemicalDatabaseIndexer(db_path=db_path, suppress_chiral=True)
        stats = indexer.index_molecules(records)
        indexer.close()

        # Because primary key is (formula, inchikey14) and chiral flags are stripped,
        # both produce identical 2D InChIKey14 and formula, resulting in 1 entry
        assert stats["total_in_db"] == 1


# ===========================================================================
# 5. ChemicalDatabaseRepository Tests
# ===========================================================================

class TestChemicalDatabaseRepository:
    """Test runtime retrieval repository, chunked search, and read pragmas."""

    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "repo_test.db"
        smiles_list = [item[1] for item in SYNTHETIC_MOLECULES]
        with ChemicalDatabaseIndexer(db_path=db_path) as idx:
            idx.index_from_smiles_list(smiles_list)
        return db_path

    def test_search_by_formulas_returns_metadata_and_fps(self, populated_db: Path):
        with ChemicalDatabaseRepository(populated_db) as repo:
            formulas = ["C9H8O4", "C6H6O"]  # Aspirin and Phenol
            meta, fps = repo.search_by_formulas(formulas, deduplicate=True)

            assert len(meta) == 2
            assert fps.shape == (2, 32)
            assert fps.dtype == np.uint64

            found_formulas = {m[0] for m in meta}
            assert found_formulas == {"C9H8O4", "C6H6O"}

    def test_search_by_formulas_chunking_over_250(self, populated_db: Path):
        """Verify formula search handles > 250 queries via chunking without SQL syntax limits."""
        with ChemicalDatabaseRepository(populated_db) as repo:
            # Query with 300 formulas containing Aspirin, Caffeine, and 298 non-existent formulas
            queries = [f"NONEXISTENT_{i}" for i in range(298)]
            queries.insert(10, "C9H8O4")
            queries.insert(260, "C8H10N4O2")

            meta, fps = repo.search_by_formulas(queries, deduplicate=True)
            assert len(meta) == 2
            assert fps.shape == (2, 32)
            retrieved_formulas = {m[0] for m in meta}
            assert retrieved_formulas == {"C9H8O4", "C8H10N4O2"}

    def test_search_by_formulas_empty_and_missing(self, populated_db: Path):
        with ChemicalDatabaseRepository(populated_db) as repo:
            meta, fps = repo.search_by_formulas([])
            assert meta == []
            assert fps.shape == (0, 32)

            meta, fps = repo.search_by_formulas(["COMPLETELY_UNKNOWN_FORMULA"])
            assert meta == []
            assert fps.shape == (0, 32)

    def test_repository_metadata_and_count(self, populated_db: Path):
        with ChemicalDatabaseRepository(populated_db) as repo:
            assert repo.count() == len(SYNTHETIC_MOLECULES)
            meta = repo.get_metadata()
            assert meta.get("candidate_count") == str(len(SYNTHETIC_MOLECULES))
            assert meta.get("schema_version") == "1.0"

    def test_in_memory_connection(self):
        indexer = ChemicalDatabaseIndexer(db_path=":memory:")
        indexer.index_from_smiles_list(["c1ccccc1"])

        repo = ChemicalDatabaseRepository(indexer.conn)
        meta, fps = repo.search_by_formulas(["C6H6"])
        assert len(meta) == 1
        assert fps.shape == (1, 32)
        indexer.close()


# ===========================================================================
# 6. SoftDatabaseSearcher Dependency Injection Tests
# ===========================================================================

class TestSoftDatabaseSearcherIntegration:
    """Test SoftDatabaseSearcher wrapped around SQLite repository vs legacy dict."""

    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "searcher_test.db"
        smiles_list = [item[1] for item in SYNTHETIC_MOLECULES]
        with ChemicalDatabaseIndexer(db_path=db_path) as idx:
            idx.index_from_smiles_list(smiles_list)
        return db_path

    def test_searcher_with_sqlite_db_path(self, populated_db: Path):
        searcher = SoftDatabaseSearcher(db_path=populated_db)
        assert searcher.repo is not None

        # Create query fingerprint matching Aspirin
        aspirin_fp = compute_morgan_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        assert aspirin_fp is not None

        candidates = searcher.search_formulas(["C9H8O4", "C7H6O2"], query_fp=aspirin_fp)
        assert len(candidates) >= 1
        top = candidates[0]
        assert isinstance(top, DBCandidate)
        assert top.source_tier == "track2_db"
        assert top.smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert top.tanimoto_score == pytest.approx(1.0, abs=1e-4)

    def test_searcher_in_memory_dict_fallback(self):
        """Verify 100% backward compatibility with legacy dictionary-based searcher."""
        mock_fp = np.zeros(2048, dtype=np.float32)
        mock_fp[10] = 1.0
        mock_db = {
            "C9H8O4": [("CC(=O)Oc1ccccc1C(=O)O", "IK14ASPIRIN001", mock_fp)],
        }
        searcher = SoftDatabaseSearcher(db=mock_db)
        assert searcher.repo is None

        query_fp = np.zeros(2048, dtype=np.float32)
        query_fp[10] = 1.0

        candidates = searcher.search_formulas(["C9H8O4"], query_fp=query_fp)
        assert len(candidates) == 1
        assert candidates[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert candidates[0].tanimoto_score == pytest.approx(1.0, abs=1e-4)

    def test_searcher_fallback_scaffolds_on_empty(self, populated_db: Path):
        fallback_fp = np.zeros(32, dtype=np.uint64)
        fallbacks = [("c1ccccc1", "IK14BENZENE001", fallback_fp)]

        searcher = SoftDatabaseSearcher(db_path=populated_db, fallback_scaffolds=fallbacks)
        query_fp = np.zeros(32, dtype=np.uint64)

        # Formula not in database
        candidates = searcher.search_formulas(["C99H99O99"], query_fp=query_fp)
        assert len(candidates) == 1
        assert candidates[0].source_tier == "track2_fallback"
        assert candidates[0].smiles == "c1ccccc1"
