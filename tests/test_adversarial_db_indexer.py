"""
Adversarial stress-test suite for SQLite chemical database indexer, repository,
popcount Tanimoto engine, and SoftDatabaseSearcher.

Covers:
1. Schema & Concurrency Stress (WITHOUT ROWID duplicates, corruption, SQL injection)
2. Popcount Tanimoto Engine Stress (extreme bit patterns, numerical bounds, latency <2ms)
3. Dead-Letter & Quarantine Stress (bizarre SMILES, unicode, binary junk, zero crashes)
4. Massive Formula Batch Querying (>1,000 formulas, >250 chunks, parameter limits)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Tuple

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
    _popcount64,
    _NUMBA_AVAILABLE,
)
from src.data.indexer import validate_chemical_database, run_tanimoto_benchmark
from src.retrieval.database_search import DBCandidate, SoftDatabaseSearcher


# ===========================================================================
# 1. Schema & Concurrency Adversarial Tests
# ===========================================================================

class TestAdversarialSchemaAndConcurrency:
    """Adversarial stress tests for SQLite WITHOUT ROWID schema and resilience."""

    def test_high_volume_duplicate_primary_keys(self, tmp_path: Path):
        """Stress WITHOUT ROWID table with 10,000 duplicate insertions of identical (formula, inchikey14)."""
        db_path = tmp_path / "dup_stress.db"
        # Aspirin: C9H8O4, InChIKey14: BSYNRYMUTXBXSQ
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        
        # 10,000 identical entries
        dup_records = [{"id": f"MOL_{i}", "smiles": aspirin_smiles} for i in range(10000)]
        
        with ChemicalDatabaseIndexer(db_path=db_path) as indexer:
            stats = indexer.index_molecules(dup_records, batch_size=2000)
            
        assert stats["indexed"] == 10000
        assert stats["failed"] == 0
        # Exactly 1 unique entry in database due to PRIMARY KEY (formula, inchikey14) WITHOUT ROWID
        assert stats["total_in_db"] == 1
        
        # Verify repository reads it cleanly
        with ChemicalDatabaseRepository(db_path) as repo:
            assert repo.count() == 1
            meta, fps = repo.search_by_formulas(["C9H8O4"])
            assert len(meta) == 1
            assert meta[0][0] == "C9H8O4"
            assert meta[0][1] == "BSYNRYMUTXBXSQ"
            assert fps.shape == (1, 32)

    def test_corrupted_and_empty_sqlite_files(self, tmp_path: Path):
        """Verify handling of 0-byte, truncated, and corrupted SQLite files."""
        # 1. Empty file (0 bytes)
        empty_file = tmp_path / "empty.db"
        empty_file.write_bytes(b"")
        
        valid, msgs = validate_chemical_database(empty_file)
        assert not valid
        assert any("not found" in m.lower() or "error" in m.lower() for m in msgs)
        
        # Opening repository on empty file in mode=ro raises or errors cleanly on query
        try:
            with ChemicalDatabaseRepository(empty_file) as repo:
                meta, fps = repo.search_by_formulas(["C6H6"])
                assert len(meta) == 0
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass  # Expected SQLite error on empty DB in mode=ro

        # 2. Corrupted file (random garbage bytes)
        corrupt_file = tmp_path / "corrupt.db"
        corrupt_file.write_bytes(b"SQLite format 3\x00" + b"\xFF\xFE\x00\x12" * 50)
        
        valid_c, msgs_c = validate_chemical_database(corrupt_file)
        assert not valid_c

    def test_sql_injection_resistance_in_formula_queries(self, tmp_path: Path):
        """Adversarial attack attempting SQL injection via malicious formula strings."""
        db_path = tmp_path / "injection_test.db"
        with ChemicalDatabaseIndexer(db_path=db_path) as indexer:
            indexer.index_from_smiles_list(["c1ccccc1", "c1ccccc1O"])
            
        repo = ChemicalDatabaseRepository(db_path)
        
        malicious_formulas = [
            "C6H6' OR '1'='1",
            "C6H6'; DROP TABLE candidates; --",
            "C6H6' UNION SELECT formula, inchikey14, smiles, fingerprint FROM candidates --",
            "'; VACUUM; --",
            "C6H6\\x00AND 1=1",
            '\" OR \"\"=\"',
            '[]\'\"\\n\\r\\t;',
            "<script>alert(1)</script>",
            "C6H6'/*comment*/--",
        ]
        
        # Repository should treat all of these as literal search keys without syntax error or table drop
        meta, fps = repo.search_by_formulas(malicious_formulas)
        assert isinstance(meta, list)
        assert isinstance(fps, np.ndarray)
        
        # Verify table still exists and has original records
        assert repo.count() == 2
        meta_valid, fps_valid = repo.search_by_formulas(["C6H6"])
        assert len(meta_valid) == 1
        repo.close()

    def test_schema_validator_catches_missing_without_rowid(self, tmp_path: Path):
        """Verify validate_chemical_database detects non-WITHOUT ROWID schema."""
        db_path = tmp_path / "regular_rowid.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE candidates (
                formula TEXT,
                inchikey14 TEXT,
                smiles TEXT,
                fingerprint BLOB,
                PRIMARY KEY (formula, inchikey14)
            );
            """
        )
        conn.execute("CREATE TABLE _schema_metadata (key TEXT PRIMARY KEY, value TEXT);")
        conn.execute("INSERT INTO _schema_metadata VALUES ('schema_version', '1.0');")
        conn.commit()
        conn.close()

        valid, msgs = validate_chemical_database(db_path)
        assert not valid
        assert any("WITHOUT ROWID" in m for m in msgs)


# ===========================================================================
# 2. Popcount Tanimoto Engine Adversarial Stress Tests
# ===========================================================================

class TestAdversarialPopcountTanimoto:
    """Stress-testing popcount Tanimoto scoring against edge cases and performance targets."""

    def test_bit_twiddling_popcount64_oracle(self):
        """Verify branchless _popcount64 matches Python bit_count for edge-case 64-bit words."""
        edge_words = [
            0,
            1,
            2,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
            0x00000000FFFFFFFF,
            0xFFFFFFFF00000000,
            0x0000FFFF0000FFFF,
        ]
        # Single bit set at each position 0..63
        edge_words.extend([1 << b for b in range(64)])
        
        # 10,000 random 64-bit words
        rng = np.random.default_rng(777)
        rand_words = rng.integers(0, np.iinfo(np.uint64).max, size=10000, dtype=np.uint64)
        edge_words.extend([int(x) for x in rand_words])
        
        for w in edge_words:
            expected = w.bit_count()
            actual = _popcount64(np.uint64(w))
            assert actual == expected, f"Failed for {hex(w)}: expected {expected}, got {actual}"

    def test_candidate_counts_scaling(self):
        """Verify popcount_tanimoto handles 0, 1, 10,000, and 50,000 candidates accurately."""
        q = np.ones(32, dtype=np.uint64) * np.uint64(0xAAAAAAAAAAAAAAAA)
        
        # 0 candidates
        empty = np.empty((0, 32), dtype=np.uint64)
        s_0 = popcount_tanimoto(q, empty)
        assert s_0.shape == (0,)
        assert s_0.dtype == np.float32

        # 1 candidate
        one = np.ones((1, 32), dtype=np.uint64) * np.uint64(0xAAAAAAAAAAAAAAAA)
        s_1 = popcount_tanimoto(q, one)
        assert s_1.shape == (1,)
        assert s_1[0] == pytest.approx(1.0, abs=1e-5)

        # 10,000 candidates
        rng = np.random.default_rng(101)
        c_10k = rng.integers(0, np.iinfo(np.uint64).max, size=(10000, 32), dtype=np.uint64)
        s_10k = popcount_tanimoto(q, c_10k)
        assert s_10k.shape == (10000,)
        assert np.all((s_10k >= 0.0) & (s_10k <= 1.0))

        # 50,000 candidates
        c_50k = rng.integers(0, np.iinfo(np.uint64).max, size=(50000, 32), dtype=np.uint64)
        s_50k = popcount_tanimoto(q, c_50k)
        assert s_50k.shape == (50000,)
        assert np.all((s_50k >= 0.0) & (s_50k <= 1.0))

    def test_extreme_fingerprint_bit_patterns(self):
        """Verify extreme bit patterns: all zeros, all ones, complementary, ultra-sparse."""
        all_zeros = np.zeros(32, dtype=np.uint64)
        all_ones = np.ones(32, dtype=np.uint64) * np.uint64(0xFFFFFFFFFFFFFFFF)
        alt_a = np.ones(32, dtype=np.uint64) * np.uint64(0xAAAAAAAAAAAAAAAA)
        alt_5 = np.ones(32, dtype=np.uint64) * np.uint64(0x5555555555555555)
        
        # Sparse: exactly 1 bit set at bit 0 of word 0
        sparse_1 = np.zeros(32, dtype=np.uint64)
        sparse_1[0] = np.uint64(1)
        # Sparse: exactly 1 bit set at bit 1 of word 0
        sparse_2 = np.zeros(32, dtype=np.uint64)
        sparse_2[0] = np.uint64(2)

        cands = np.stack([all_zeros, all_ones, alt_a, alt_5, sparse_1, sparse_2], axis=0)

        # 1. Query: all zeros -> all similarities 0.0
        scores_z = popcount_tanimoto(all_zeros, cands)
        np.testing.assert_allclose(scores_z, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 2. Query: all ones
        scores_o = popcount_tanimoto(all_ones, cands)
        # Tanimoto(all_ones, all_zeros) = 0.0
        # Tanimoto(all_ones, all_ones) = 2048 / 2048 = 1.0
        # Tanimoto(all_ones, alt_a) = 1024 / 2048 = 0.5
        # Tanimoto(all_ones, alt_5) = 1024 / 2048 = 0.5
        # Tanimoto(all_ones, sparse_1) = 1 / 2048 = 0.00048828
        assert scores_o[0] == 0.0
        assert scores_o[1] == pytest.approx(1.0, abs=1e-5)
        assert scores_o[2] == pytest.approx(0.5, abs=1e-5)
        assert scores_o[3] == pytest.approx(0.5, abs=1e-5)
        assert scores_o[4] == pytest.approx(1.0 / 2048.0, abs=1e-5)

        # 3. Query: alt_a (even bits) vs alt_5 (odd bits) -> 0 overlap
        scores_a = popcount_tanimoto(alt_a, cands)
        assert scores_a[2] == pytest.approx(1.0, abs=1e-5)  # Self-match
        assert scores_a[3] == 0.0                           # Disjoint

        # 4. Disjoint sparse
        scores_s1 = popcount_tanimoto(sparse_1, cands)
        assert scores_s1[4] == pytest.approx(1.0, abs=1e-5)  # sparse_1 vs sparse_1
        assert scores_s1[5] == 0.0                           # sparse_1 vs sparse_2

    def test_numerical_stability_and_bounds_assertion(self):
        """Assert strict bounds [0.0, 1.0], no NaNs, no Infs across 10,000 random comparisons."""
        rng = np.random.default_rng(42)
        q = rng.integers(0, np.iinfo(np.uint64).max, size=32, dtype=np.uint64)
        cands = rng.integers(0, np.iinfo(np.uint64).max, size=(10000, 32), dtype=np.uint64)

        scores = popcount_tanimoto(q, cands)

        assert not np.isnan(scores).any(), "Found NaN in popcount Tanimoto scores"
        assert not np.isinf(scores).any(), "Found Inf in popcount Tanimoto scores"
        assert (scores >= 0.0).all(), "Found score < 0.0"
        assert (scores <= 1.0).all(), "Found score > 1.0"

    def test_mathematical_equivalence_against_numpy_oracle(self):
        """Verify Numba JIT output exactly matches NumPy 2.x bitwise_count across 10,000 pairs."""
        rng = np.random.default_rng(999)
        q = rng.integers(0, np.iinfo(np.uint64).max, size=32, dtype=np.uint64)
        cands = rng.integers(0, np.iinfo(np.uint64).max, size=(10000, 32), dtype=np.uint64)

        # 1. Calculate via popcount_tanimoto (Numba fast path if available)
        actual_scores = popcount_tanimoto(q, cands)

        # 2. Calculate via pure NumPy ground-truth oracle
        inter = np.bitwise_count(cands & q).sum(axis=1)
        cand_sums = np.bitwise_count(cands).sum(axis=1)
        q_sum = int(np.bitwise_count(q).sum())
        union = cand_sums + q_sum - inter
        expected_scores = np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)

        np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-5, atol=1e-5)

    def test_latency_benchmark_sub_2ms_rigorous(self):
        """Empirically benchmark 10,000 candidates scoring latency against < 2.0 ms limit."""
        passed, latency_ms = run_tanimoto_benchmark(num_candidates=10000, iterations=50, threshold_ms=2.0)
        assert passed, f"10k benchmark failed: latency was {latency_ms:.3f} ms, threshold is < 2.0 ms"
        assert latency_ms < 2.0


# ===========================================================================
# 3. Dead-Letter & Quarantine Stress Tests
# ===========================================================================

class TestAdversarialDeadLetterQuarantine:
    """Adversarial inputs to stress dead-letter logging and crash resistance."""

    def test_bizarre_and_corrupted_smiles_quarantine(self, tmp_path: Path):
        """Feed chemically impossible, malformed, non-ASCII, and binary junk SMILES."""
        db_path = tmp_path / "quarantine_stress.db"
        dead_letter_log = tmp_path / "dead_letters_stress.jsonl"

        corrupted_inputs = [
            # Hypervalent carbons
            {"id": "HYPER_C1", "smiles": "C(C)(C)(C)(C)C"},
            {"id": "HYPER_C2", "smiles": "c1ccccc1(C)(C)(C)(C)"},
            {"id": "HYPER_N", "smiles": "N(=O)(=O)(=O)(=O)=O"},
            # Unclosed rings & broken syntax
            {"id": "UNCLOSED_1", "smiles": "C1CCCCC"},
            {"id": "UNCLOSED_2", "smiles": "c1ccccc"},
            {"id": "BROKEN_PAREN", "smiles": "CC((C)O"},
            {"id": "EXTRA_PAREN", "smiles": "CC(C))O"},
            # Unicode, emoji, and control chars
            {"id": "EMOJI", "smiles": "c1ccccc1\U0001F9EA\U0001F9EC"},
            {"id": "NON_ASCII", "smiles": "C(=O)Oéèà"},
            {"id": "NULL_BYTE", "smiles": "c1ccccc1\x00O"},
            {"id": "WHITESPACE_ONLY", "smiles": "    \t\n  "},
            {"id": "EMPTY_STR", "smiles": ""},
            {"id": "NONE_VAL", "smiles": None},
            # Non-chemical text
            {"id": "RANDOM_WORDS", "smiles": "THIS_IS_NOT_CHEMISTRY"},
            {"id": "INCHI_STR", "smiles": "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"},
            # Extreme length string (nested parens)
            {"id": "DEEP_NESTING", "smiles": "C" + "("*100 + "C" + ")"*100},
        ]

        # Add valid controls interleaved
        valid_controls = [
            {"id": "VALID_ASPIRIN", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"id": "VALID_BENZENE", "smiles": "c1ccccc1"},
            {"id": "VALID_PHENOL", "smiles": "c1ccccc1O"},
        ]

        # Combine interleaved
        all_records = []
        for i, bad in enumerate(corrupted_inputs):
            all_records.append(bad)
            if i < len(valid_controls):
                all_records.append(valid_controls[i])

        with ChemicalDatabaseIndexer(
            db_path=db_path,
            dead_letter_path=dead_letter_log,
        ) as indexer:
            stats = indexer.index_molecules(all_records, batch_size=5)

        # Zero crashes
        assert stats["indexed"] == 3
        assert stats["failed"] == len(corrupted_inputs)
        assert stats["total"] == len(all_records)
        assert stats["total_in_db"] == 3

        # Verify dead letter file
        assert dead_letter_log.exists()
        dead_lines = dead_letter_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(dead_lines) == len(corrupted_inputs)

        # Verify each dead letter has id, smiles, reason, timestamp
        for line in dead_lines:
            entry = json.loads(line)
            assert "id" in entry
            assert "smiles" in entry
            assert "reason" in entry
            assert "timestamp" in entry


# ===========================================================================
# 4. Massive Formula Batch Querying Tests
# ===========================================================================

class TestAdversarialMassiveBatchQueries:
    """Stress-test formula chunking across >1,000 formulas (>4 chunks of 250)."""

    def test_chunking_1250_formulas(self, tmp_path: Path):
        """Query 1,250 formulas (5 full chunks of 250) against SQLite repository."""
        db_path = tmp_path / "massive_batch.db"

        # Index 20 known compounds
        known_smiles = [
            ("C9H8O4", "CC(=O)Oc1ccccc1C(=O)O"),     # Aspirin
            ("C8H10N4O2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),  # Caffeine
            ("C6H6", "c1ccccc1"),                    # Benzene
            ("C7H6O2", "c1ccccc1C(=O)O"),            # Benzoic acid
            ("C6H6O", "c1ccccc1O"),                  # Phenol
        ]
        with ChemicalDatabaseIndexer(db_path=db_path) as indexer:
            indexer.index_from_smiles_list([s[1] for s in known_smiles])

        # Construct 1,250 queries (5 known formulas scattered among 1,245 random bogus formulas)
        bogus_formulas = [f"BOGUS_FORMULA_{i:04d}" for i in range(1245)]
        
        all_formulas = []
        # Insert known formulas at various positions across chunks
        all_formulas.extend(bogus_formulas[:100])
        all_formulas.append("C9H8O4")       # Chunk 0
        all_formulas.extend(bogus_formulas[100:350])
        all_formulas.append("C8H10N4O2")    # Chunk 1
        all_formulas.extend(bogus_formulas[350:650])
        all_formulas.append("C6H6")         # Chunk 2
        all_formulas.extend(bogus_formulas[650:950])
        all_formulas.append("C7H6O2")       # Chunk 3
        all_formulas.extend(bogus_formulas[950:])
        all_formulas.append("C6H6O")        # Chunk 4

        assert len(all_formulas) == 1250

        with ChemicalDatabaseRepository(db_path) as repo:
            # Query with deduplicate=True
            meta, fps = repo.search_by_formulas(all_formulas, deduplicate=True)
            assert len(meta) == 5
            assert fps.shape == (5, 32)
            retrieved_formulas = {m[0] for m in meta}
            assert retrieved_formulas == {"C9H8O4", "C8H10N4O2", "C6H6", "C7H6O2", "C6H6O"}

    def test_soft_database_searcher_massive_batch(self, tmp_path: Path):
        """Verify SoftDatabaseSearcher integration with 1,250 formulas and 2048-dim float query."""
        db_path = tmp_path / "searcher_massive.db"
        with ChemicalDatabaseIndexer(db_path=db_path) as indexer:
            indexer.index_from_smiles_list(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"])

        searcher = SoftDatabaseSearcher(db_path=db_path)

        # 2048-dim float query matching Aspirin
        aspirin_fp = compute_morgan_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        # 1,250 formulas
        formulas = [f"FAKE_{i}" for i in range(1249)] + ["C9H8O4"]

        candidates = searcher.search_formulas(formulas, query_fp=aspirin_fp)
        assert len(candidates) >= 1
        assert candidates[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert candidates[0].tanimoto_score == pytest.approx(1.0, abs=1e-4)
        assert candidates[0].source_tier == "track2_db"
        searcher.close()
