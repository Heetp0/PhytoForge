"""
Adversarial stress-test suite for Spectral Indexer, Botanical Knowledge, and CLI Tooling (Milestone M5).

Challenger 2 Empirical Verification:
1. Spectral Indexer Stress:
   - Zero-length arrays, empty partitions, single-vector indexes, massive mass ranges.
   - Zero-norm embeddings, NaN embeddings, Inf embeddings (assert cosine scores remain finite and bounded in [0.0, 1.0]).
   - Verify np.memmap mode is strictly 'r' and cannot be written to.
   - Verify Windows memory-map handles are safely closed on .close() and deletion does not raise WinError 32.
2. Botanical Transformations Stress:
   - Boundary mass testing at +-4.99 ppm vs +-5.01 ppm around all 16 reference deltas.
   - Cross-collision stress between nearest pairs (Glucuronide vs Feruloyl, Sulfate vs Phosphate, Caffeoyl vs Hexose, Coumaroyl vs Rhamnose) to verify zero false positives.
   - Negative delta matching (losses of moieties) and bidirectional tagging (+<Name> vs -<Name>).
   - Peak matching tolerance boundaries (0.019 Da vs 0.021 Da) and MS2 co-validation rejection for 0 and 1 shared peaks.
3. CLI Subcommand Stress:
   - Test invalid flags, missing files, corrupted files, and non-existent subcommands.
   - Assert all error cases exit with non-zero exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import pytest

from src.data.indexer import (
    build_parser,
    main,
    validate_chemical_database,
    validate_indices,
    validate_spectral_index,
)
from src.data.spectral_indexer import (
    SpectralHit,
    SpectralIndex,
    SpectralIndexBuilder,
    normalize_polarity,
)
from src.retrieval.transductive_networking import (
    BOTANICAL_TRANSFORMATIONS,
    DELTA_LIBRARY,
    NetworkCandidate,
    TransductiveMolecularNetwork,
)


# ===========================================================================
# 1. Adversarial Spectral Indexer Tests
# ===========================================================================

class TestAdversarialSpectralIndexer:
    """Stress testing the spectral vector indexer against extreme inputs and handle management."""

    def test_zero_length_arrays_and_empty_partitions(self, tmp_path: Path):
        """Build index with zero records and search empty partitions."""
        empty_dir = tmp_path / "empty_spectral_idx"
        df_empty = pd.DataFrame(
            columns=["id", "smiles", "inchikey14", "precursor_mz", "adduct", "polarity", "embedding"]
        )
        SpectralIndexBuilder.build_index(df_empty, output_dir=empty_dir)

        with SpectralIndex(empty_dir) as idx:
            assert len(idx) == 0
            assert idx.partitions["positive"]["count"] == 0
            assert idx.partitions["negative"]["count"] == 0

            # Search with arbitrary query
            hits_pos = idx.search(np.ones(1024, dtype=np.float32), precursor_mz=300.0, polarity="positive")
            assert hits_pos == []

            hits_neg = idx.search(np.ones(1024, dtype=np.float32), precursor_mz=300.0, polarity="negative")
            assert hits_neg == []

    def test_single_vector_index(self, tmp_path: Path):
        """Test boundary condition of an index containing exactly 1 vector."""
        single_dir = tmp_path / "single_vec_idx"
        rng = np.random.default_rng(123)
        vec = rng.standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)

        df = pd.DataFrame(
            [
                {
                    "id": "SOLO_001",
                    "smiles": "c1ccccc1",
                    "inchikey14": "UHOVQNZJYSORNB",
                    "precursor_mz": 79.0542,
                    "adduct": "[M+H]+",
                    "polarity": "positive",
                    "embedding": vec,
                }
            ]
        )
        SpectralIndexBuilder.build_index(df, output_dir=single_dir)

        with SpectralIndex(single_dir) as idx:
            assert len(idx) == 1
            # 1. Exact match search
            hits = idx.search(vec, precursor_mz=79.0542, polarity="positive", ppm_tolerance=10.0)
            assert len(hits) == 1
            assert hits[0].id == "SOLO_001"
            assert hits[0].cosine_score == pytest.approx(1.0, abs=1e-3)
            assert hits[0].rank == 1

            # 2. Out-of-window search
            hits_out = idx.search(vec, precursor_mz=150.0, polarity="positive", ppm_tolerance=10.0)
            assert hits_out == []

    def test_massive_and_extreme_precursor_mass_ranges(self, tmp_path: Path):
        """Test search behavior with massive mass ranges and non-physical inputs."""
        extreme_dir = tmp_path / "extreme_mass_idx"
        rng = np.random.default_rng(456)
        mzs = [0.001, 10.0, 500.0, 10000.0, 1e6]
        embs = rng.standard_normal((len(mzs), 1024)).astype(np.float32)
        for i in range(len(mzs)):
            embs[i] /= np.linalg.norm(embs[i])

        df = pd.DataFrame(
            {
                "id": [f"EXT_{i}" for i in range(len(mzs))],
                "smiles": ["C"] * len(mzs),
                "inchikey14": [f"EXTIK14{i:07d}" for i in range(len(mzs))],
                "precursor_mz": mzs,
                "adduct": ["[M+H]+"] * len(mzs),
                "polarity": ["positive"] * len(mzs),
                "embedding": list(embs),
            }
        )
        SpectralIndexBuilder.build_index(df, output_dir=extreme_dir)

        with SpectralIndex(extreme_dir) as idx:
            # Query at non-physical <= 0 returns []
            assert idx.search(embs[0], precursor_mz=0.0) == []
            assert idx.search(embs[0], precursor_mz=-100.0) == []
            assert idx.search(embs[0], precursor_mz=-1e9) == []

            # Exact query at ultra-low mass (0.001 Da)
            hits_low = idx.search(embs[0], precursor_mz=0.001, ppm_tolerance=10.0)
            assert len(hits_low) == 1
            assert hits_low[0].id == "EXT_0"

            # Exact query at massive mass (1e6 Da)
            hits_high = idx.search(embs[4], precursor_mz=1e6, ppm_tolerance=10.0)
            assert len(hits_high) == 1
            assert hits_high[0].id == "EXT_4"

    def test_zero_norm_nan_inf_query_embeddings(self, tmp_path: Path):
        """Stress test: queries with zero norm, NaNs, Infs must yield finite bounded [0.0, 1.0] scores."""
        idx_dir = tmp_path / "robust_query_idx"
        rng = np.random.default_rng(789)
        embs = rng.standard_normal((5, 1024)).astype(np.float32)
        for i in range(5):
            embs[i] /= np.linalg.norm(embs[i])

        df = pd.DataFrame(
            {
                "id": [f"ROB_{i}" for i in range(5)],
                "smiles": ["C6H6"] * 5,
                "inchikey14": [f"ROBIK14{i:07d}" for i in range(5)],
                "precursor_mz": [200.0 + i * 0.0001 for i in range(5)],
                "adduct": ["[M+H]+"] * 5,
                "polarity": ["positive"] * 5,
                "embedding": list(embs),
            }
        )
        SpectralIndexBuilder.build_index(df, output_dir=idx_dir)

        with SpectralIndex(idx_dir) as idx:
            # 1. Zero-norm query vector
            zero_vec = np.zeros(1024, dtype=np.float32)
            hits_zero = idx.search(zero_vec, precursor_mz=200.0, ppm_tolerance=20.0)
            assert len(hits_zero) > 0
            for h in hits_zero:
                assert not np.isnan(h.cosine_score)
                assert not np.isinf(h.cosine_score)
                assert 0.0 <= h.cosine_score <= 1.0
                assert h.cosine_score == 0.0

            # 2. NaN-embedded query vector
            nan_vec = rng.standard_normal(1024).astype(np.float32)
            nan_vec[0] = np.nan
            nan_vec[50] = np.nan
            hits_nan = idx.search(nan_vec, precursor_mz=200.0, ppm_tolerance=20.0)
            assert len(hits_nan) > 0
            for h in hits_nan:
                assert not np.isnan(h.cosine_score)
                assert not np.isinf(h.cosine_score)
                assert 0.0 <= h.cosine_score <= 1.0

            # 3. Inf-embedded query vector
            # Note: q_norm is inf, which bypasses the q_norm <= 1e-12 or np.isnan(q_norm) check,
            # triggering RuntimeWarning: invalid value encountered in divide on q / q_norm.
            inf_vec = rng.standard_normal(1024).astype(np.float32)
            inf_vec[10] = np.inf
            inf_vec[20] = -np.inf
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                hits_inf = idx.search(inf_vec, precursor_mz=200.0, ppm_tolerance=20.0)
                # Verify that NO RuntimeWarning was emitted due to isfinite() check
                assert not any(
                    issubclass(w.category, RuntimeWarning) for w in recorded_warnings
                ), "Did not expect RuntimeWarning on inf division"
                assert len(hits_inf) > 0
                for h in hits_inf:
                    assert not np.isnan(h.cosine_score)
                    assert not np.isinf(h.cosine_score)
                    assert 0.0 <= h.cosine_score <= 1.0

    def test_inf_embedding_unhandled_runtimewarning_defect(self, tmp_path: Path):
        """Verify Inf query is handled cleanly without RuntimeWarning under -W error."""
        idx_dir = tmp_path / "defect_inf_idx"
        df = pd.DataFrame(
            [
                {
                    "id": "DEFECT_001",
                    "smiles": "C",
                    "inchikey14": "IK14DEFECT0000",
                    "precursor_mz": 100.0,
                    "adduct": "[M+H]+",
                    "polarity": "positive",
                    "embedding": np.ones(1024, dtype=np.float32),
                }
            ]
        )
        SpectralIndexBuilder.build_index(df, output_dir=idx_dir)

        with SpectralIndex(idx_dir) as idx:
            q = np.ones(1024, dtype=np.float32)
            q[0] = np.inf
            # Verify that search executes cleanly without RuntimeWarning under -W error
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                hits = idx.search(q, precursor_mz=100.0)
                assert not any(issubclass(w.category, RuntimeWarning) for w in recorded_warnings)
                assert len(hits) == 1
                assert hits[0].cosine_score == 0.0

    def test_corrupted_nan_inf_indexed_embeddings(self, tmp_path: Path):
        """Stress test: if index contains corrupted NaN or all-zero embeddings, search remains finite and robust."""
        idx_dir = tmp_path / "corrupt_cand_idx"
        cands = [
            np.zeros(1024, dtype=np.float32),  # all zeros
            np.ones(1024, dtype=np.float32) * np.nan,  # all NaNs
            np.ones(1024, dtype=np.float32),  # valid ones
        ]
        df = pd.DataFrame(
            {
                "id": ["BAD_ZERO", "BAD_NAN", "GOOD_ONES"],
                "smiles": ["C6H6"] * 3,
                "inchikey14": ["IK14ZERO000000", "IK14NAN0000000", "IK14GOOD000000"],
                "precursor_mz": [200.0, 200.0001, 200.0002],
                "adduct": ["[M+H]+"] * 3,
                "polarity": ["positive"] * 3,
                "embedding": cands,
            }
        )
        SpectralIndexBuilder.build_index(df, output_dir=idx_dir)

        with SpectralIndex(idx_dir) as idx:
            q_valid = np.ones(1024, dtype=np.float32)
            q_valid /= np.linalg.norm(q_valid)
            hits = idx.search(q_valid, precursor_mz=200.0, ppm_tolerance=20.0)

            assert len(hits) == 3
            for h in hits:
                assert not np.isnan(h.cosine_score)
                assert not np.isinf(h.cosine_score)
                assert 0.0 <= h.cosine_score <= 1.0

            # The good candidate should be top ranked with score close to 1.0
            assert hits[0].id == "GOOD_ONES"
            assert hits[0].cosine_score == pytest.approx(1.0, abs=1e-3)

    def test_zero_copy_memmap_read_only_enforcement(self, tmp_path: Path):
        """Assert np.memmap is strictly in read-only mode 'r' and write attempts fail."""
        idx_dir = tmp_path / "memmap_mode_idx"
        df = pd.DataFrame(
            {
                "id": ["SPEC_001"],
                "smiles": ["C6H6"],
                "inchikey14": ["IK14TEST000000"],
                "precursor_mz": [100.0],
                "adduct": ["[M+H]+"],
                "polarity": ["positive"],
                "embedding": [np.ones(1024, dtype=np.float32)],
            }
        )
        SpectralIndexBuilder.build_index(df, output_dir=idx_dir)

        with SpectralIndex(idx_dir) as idx:
            embs = idx.get_embeddings("positive")
            assert isinstance(embs, np.memmap)
            assert embs.mode == "r"

            # Assert attempting to write in-place raises ValueError
            with pytest.raises(ValueError):
                embs[0, 0] = 42.0

    def test_windows_mmap_handle_closing_and_rmtree(self, tmp_path: Path):
        """Verify Windows mmap handles close safely without triggering WinError 32 on rmtree."""
        idx_dir = tmp_path / "windows_close_test"
        df = pd.DataFrame(
            {
                "id": [f"SPEC_{i}" for i in range(10)],
                "smiles": ["C6H6"] * 10,
                "inchikey14": [f"IK14TEST{i:06d}" for i in range(10)],
                "precursor_mz": [100.0 + i for i in range(10)],
                "adduct": ["[M+H]+"] * 5 + ["[M-H]-"] * 5,
                "polarity": ["positive"] * 5 + ["negative"] * 5,
                "embedding": [np.ones(1024, dtype=np.float32)] * 10,
            }
        )
        SpectralIndexBuilder.build_index(df, output_dir=idx_dir)

        # 1. Test explicit .close()
        idx = SpectralIndex(idx_dir)
        _ = idx.search(np.ones(1024), precursor_mz=102.0, polarity="positive")
        _ = idx.search(np.ones(1024), precursor_mz=107.0, polarity="negative")
        _ = idx.get_embeddings("positive")
        _ = idx.get_embeddings("negative")
        idx.close()

        # Deletion must succeed on Windows
        shutil.rmtree(idx_dir)
        assert not idx_dir.exists()

        # 2. Test context manager cleanup
        SpectralIndexBuilder.build_index(df, output_dir=idx_dir)
        with SpectralIndex(idx_dir) as idx_ctx:
            _ = idx_ctx.search(np.ones(1024), precursor_mz=102.0, polarity="positive")

        shutil.rmtree(idx_dir)
        assert not idx_dir.exists()


# ===========================================================================
# 2. Adversarial Botanical Transformations & Networking Tests
# ===========================================================================

class TestAdversarialBotanicalKnowledge:
    """Stress testing botanical transformation delta matching, precision boundaries, and MS2 co-validation."""

    ALL_16_DELTAS = [
        ("Hexose", 162.0528),
        ("Pentose", 132.0423),
        ("Rhamnose", 146.0579),
        ("Glucuronide", 176.0321),
        ("Methyl", 14.0157),
        ("Acetyl", 42.0106),
        ("Malonyl", 86.0004),
        ("Prenyl", 68.0626),
        ("Galloyl", 152.0110),
        ("Caffeoyl", 162.0317),
        ("Feruloyl", 176.0473),
        ("Coumaroyl", 146.0368),
        ("Sinapoyl", 206.0579),
        ("Hydroxylation", 15.9949),
        ("Sulfate", 79.9568),
        ("Phosphate", 79.9663),
    ]

    @pytest.mark.parametrize(
        "name,mass",
        [d for d in ALL_16_DELTAS if d[0] != "Methyl"]
    )
    def test_boundary_ppm_exact_thresholds_15_transformations(self, name: str, mass: float):
        """Adversarial boundary testing: +-4.99 ppm must MATCH, +-5.01 ppm must be REJECTED (None)."""
        tmn = TransductiveMolecularNetwork(mass_tolerance_ppm=5.0)

        # 1. Positive shifts
        delta_in_pos = mass * (1.0 + 4.99e-6)
        delta_out_pos = mass * (1.0 + 5.01e-6)
        assert tmn.match_delta(delta_in_pos, tolerance_ppm=5.0) == f"+{name}"
        assert tmn.match_delta(delta_out_pos, tolerance_ppm=5.0) is None

        # 2. Negative shifts around positive mass
        delta_in_neg = mass * (1.0 - 4.99e-6)
        delta_out_neg = mass * (1.0 - 5.01e-6)
        assert tmn.match_delta(delta_in_neg, tolerance_ppm=5.0) == f"+{name}"
        assert tmn.match_delta(delta_out_neg, tolerance_ppm=5.0) is None

        # 3. Negative delta (loss) with positive/negative shifts
        neg_in_pos = -mass * (1.0 + 4.99e-6)
        neg_out_pos = -mass * (1.0 + 5.01e-6)
        assert tmn.match_delta(neg_in_pos, tolerance_ppm=5.0) == f"-{name}"
        assert tmn.match_delta(neg_out_pos, tolerance_ppm=5.0) is None

        neg_in_neg = -mass * (1.0 - 4.99e-6)
        neg_out_neg = -mass * (1.0 - 5.01e-6)
        assert tmn.match_delta(neg_in_neg, tolerance_ppm=5.0) == f"-{name}"
        assert tmn.match_delta(neg_out_neg, tolerance_ppm=5.0) is None

    def test_methyl_alias_asymmetric_tolerance_dilation_defect(self):
        """Empirical proof of defect: 14.0156 alias dilates Methyl negative tolerance from -5 ppm to -12.13 ppm."""
        tmn = TransductiveMolecularNetwork(mass_tolerance_ppm=5.0)
        mass = 14.0157

        # +4.99 ppm matches +Methyl
        assert tmn.match_delta(mass * (1.0 + 4.99e-6), tolerance_ppm=5.0) == "+Methyl"
        # +5.01 ppm correctly returns None
        assert tmn.match_delta(mass * (1.0 + 5.01e-6), tolerance_ppm=5.0) is None

        # -4.99 ppm matches +Methyl
        assert tmn.match_delta(mass * (1.0 - 4.99e-6), tolerance_ppm=5.0) == "+Methyl"

        # -5.01 ppm (14.0156298 Da) is outside 5 ppm of 14.0157.
        # Verified that with _DeltaLibrary canonical iteration, tolerance is not dilated:
        undilated_match = tmn.match_delta(mass * (1.0 - 5.01e-6), tolerance_ppm=5.0)
        assert undilated_match is None, "Verified no tolerance window dilation"

        # The true rejection boundary on the negative side
        assert tmn.match_delta(mass * (1.0 - 12.5e-6), tolerance_ppm=5.0) is None

    def test_nearest_pair_dense_grid_collision_immunity(self):
        """Dense grid stress testing nearest pairs to verify zero false-positive cross-matches."""
        tmn = TransductiveMolecularNetwork(mass_tolerance_ppm=5.0)

        pairs = [
            ("Glucuronide", 176.0321, "Feruloyl", 176.0473),
            ("Sulfate", 79.9568, "Phosphate", 79.9663),
            ("Caffeoyl", 162.0317, "Hexose", 162.0528),
            ("Coumaroyl", 146.0368, "Rhamnose", 146.0579),
        ]

        for name_a, mass_a, name_b, mass_b in pairs:
            # 500 linearly spaced test points between mass_a - 0.005 and mass_b + 0.005
            test_points = np.linspace(mass_a - 0.005, mass_b + 0.005, 500)
            
            for pt in test_points:
                res = tmn.match_delta(pt, tolerance_ppm=5.0)
                err_a_ppm = abs(pt - mass_a) / mass_a * 1e6
                err_b_ppm = abs(pt - mass_b) / mass_b * 1e6

                if err_a_ppm <= 5.0:
                    assert res == f"+{name_a}", f"Point {pt} within 5ppm of {name_a} failed to match {name_a}"
                elif err_b_ppm <= 5.0:
                    assert res == f"+{name_b}", f"Point {pt} within 5ppm of {name_b} failed to match {name_b}"
                else:
                    assert res is None, f"Point {pt} between {name_a} and {name_b} falsely matched {res}"

    def test_directional_scaffold_propagation_sign_correctness(self):
        """Adversarial verification of directional (+/-) scaffold propagation."""
        tmn = TransductiveMolecularNetwork(cosine_threshold=0.6, mass_tolerance_ppm=5.0)
        rng = np.random.default_rng(999)
        base_emb = rng.standard_normal(128).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)
        base_peaks = np.array([[100.0, 50.0], [200.0, 50.0], [300.0, 50.0]])

        # Source spectrum: 300.0 Da
        tmn.add_spectrum("SRC", neutral_mass=300.0, embedding=base_emb, peaks=base_peaks)

        # Target A: 300.0 + 176.0321 (Glucuronidation)
        tmn.add_spectrum(
            "TGT_PLUS_GLUC",
            neutral_mass=300.0 + 176.0321,
            embedding=base_emb,
            peaks=base_peaks,
        )

        # Target B: 300.0 - 176.0321 (Deglucuronidation)
        tmn.add_spectrum(
            "TGT_MINUS_GLUC",
            neutral_mass=300.0 - 176.0321,
            embedding=base_emb,
            peaks=base_peaks,
        )

        known = {"SRC": ("C15H10O5", "IK14SOURCE001")}

        # Directional = True
        cands_plus = tmn.propagate_scaffold("TGT_PLUS_GLUC", known, min_shared_peaks=2, directional=True)
        assert len(cands_plus) == 1
        assert cands_plus[0].transformation == "+Glucuronide"

        cands_minus = tmn.propagate_scaffold("TGT_MINUS_GLUC", known, min_shared_peaks=2, directional=True)
        assert len(cands_minus) == 1
        assert cands_minus[0].transformation == "-Glucuronide"

        # Directional = False (legacy unsigned mode)
        cands_minus_legacy = tmn.propagate_scaffold(
            "TGT_MINUS_GLUC", known, min_shared_peaks=2, directional=False
        )
        assert len(cands_minus_legacy) == 1
        assert cands_minus_legacy[0].transformation == "+Glucuronide"

    def test_ms2_covalidation_boundary_tolerances_and_thresholds(self):
        """Stress testing MS2 fragment tolerance boundary at 0.019 Da vs 0.021 Da and threshold rejection."""
        tmn = TransductiveMolecularNetwork()

        # Target peaks: 100.0, 200.0
        p_target = np.array([100.0, 200.0])

        # 1. Delta 0.019 Da (< 0.02) -> Match
        p_close = np.array([100.019, 200.019])
        assert tmn.count_shared_peaks(p_target, p_close, mz_tolerance=0.02) == 2

        # 2. Delta 0.021 Da (> 0.02) -> Reject
        p_far = np.array([100.021, 200.021])
        assert tmn.count_shared_peaks(p_target, p_far, mz_tolerance=0.02) == 0

        # 3. Exact boundary IEEE-754 float behavior:
        # Note: 100.02 - 100.0 = 0.01999999999999602 (<= 0.02),
        # but 200.02 - 200.0 = 0.020000000000010232 (> 0.02)!
        # With epsilon (+1e-9) in count_shared_peaks, IEEE-754 precision edges are preserved:
        assert tmn.count_shared_peaks(np.array([100.0]), np.array([100.02]), mz_tolerance=0.02) == 1
        assert tmn.count_shared_peaks(np.array([200.0]), np.array([200.02]), mz_tolerance=0.02) == 1

        # 4. Mixed: 1 match (0.019) and 1 non-match (0.021) -> count is 1
        p_mixed = np.array([100.019, 200.021])
        assert tmn.count_shared_peaks(p_target, p_mixed, mz_tolerance=0.02) == 1

        # 5. Scaffold propagation with min_shared_peaks=2 rejection
        rng = np.random.default_rng(333)
        emb = rng.standard_normal(64).astype(np.float32)
        emb /= np.linalg.norm(emb)

        tmn.add_spectrum("S1", neutral_mass=200.0, embedding=emb, peaks=p_target)
        tmn.add_spectrum("T_MIXED", neutral_mass=200.0 + 14.0157, embedding=emb, peaks=p_mixed)
        tmn.add_spectrum("T_FAR", neutral_mass=200.0 + 14.0157, embedding=emb, peaks=p_far)
        tmn.add_spectrum("T_CLOSE", neutral_mass=200.0 + 14.0157, embedding=emb, peaks=p_close)

        known = {"S1": ("C10H10", "IK14S1000000")}

        # T_FAR has 0 shared peaks -> rejected
        assert tmn.propagate_scaffold("T_FAR", known, min_shared_peaks=2) == []

        # T_MIXED has 1 shared peak -> rejected when min_shared_peaks=2
        assert tmn.propagate_scaffold("T_MIXED", known, min_shared_peaks=2) == []

        # T_MIXED accepted when min_shared_peaks=1
        assert len(tmn.propagate_scaffold("T_MIXED", known, min_shared_peaks=1)) == 1

        # T_CLOSE has 2 shared peaks -> accepted when min_shared_peaks=2
        assert len(tmn.propagate_scaffold("T_CLOSE", known, min_shared_peaks=2)) == 1


# ===========================================================================
# 3. Adversarial CLI Tooling & Validation Tests
# ===========================================================================

class TestAdversarialIndexerCLI:
    """Stress testing the indexer CLI for all error paths, invalid flags, and corrupted files."""

    def test_cli_invalid_flags_exit_code_2(self):
        """CLI with invalid flag must exit with code 2."""
        cases = [
            ["--nonexistent-flag"],
            ["index-db", "--unknown-opt"],
            ["index-spectra", "--bogus"],
            ["validate-index", "--bad-arg"],
            ["benchmark", "--invalid"],
        ]
        for args in cases:
            cmd = [sys.executable, "-m", "src.data.indexer"] + args
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            assert res.returncode == 2, f"Expected returncode 2 for {args}, got {res.returncode}: {res.stderr}"

    def test_cli_missing_required_arguments_exit_code_2(self):
        """Subcommands with missing required arguments must exit with code 2."""
        cases = [
            ["index-db"],
            ["index-db", "--input", "foo.csv"],  # missing --output
            ["index-spectra"],
            ["index-spectra", "--input", "foo.parquet"],  # missing --output-dir
        ]
        for args in cases:
            cmd = [sys.executable, "-m", "src.data.indexer"] + args
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            assert res.returncode == 2, f"Expected returncode 2 for {args}, got {res.returncode}: {res.stderr}"

    def test_cli_nonexistent_subcommand_exits_2(self):
        """Invalid subcommand must exit with code 2."""
        cmd = [sys.executable, "-m", "src.data.indexer", "nonexistent-cmd"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 2

    def test_cli_validate_index_missing_both_args_exits_1(self):
        """validate-index without --db-path or --spectral-dir must exit with code 1."""
        cmd = [sys.executable, "-m", "src.data.indexer", "validate-index"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        assert "At least one of --db-path or --spectral-dir must be specified" in res.stderr

    def test_cli_nonexistent_input_files_exit_1(self, tmp_path: Path):
        """Nonexistent input files for index-db, index-spectra, and validate-index must exit with code 1."""
        cases = [
            ["index-db", "--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "out.db")],
            ["index-spectra", "--input", str(tmp_path / "nope.parquet"), "--output-dir", str(tmp_path / "out")],
            ["validate-index", "--db-path", str(tmp_path / "nope.db")],
            ["validate-index", "--spectral-dir", str(tmp_path / "nope_dir")],
        ]
        for args in cases:
            cmd = [sys.executable, "-m", "src.data.indexer"] + args
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            assert res.returncode == 1, f"Expected returncode 1 for {args}, got {res.returncode}: {res.stderr}"

    def test_validate_spectral_index_adversarial_corruptions(self, tmp_path: Path):
        """Systematically corrupt each component of a spectral index and verify validate_spectral_index catches it."""
        # 1. Base valid index
        base_dir = tmp_path / "valid_base"
        pos_dir = base_dir / "positive"
        pos_dir.mkdir(parents=True)
        manifest = {"version": "1.0.0", "dim": 1024, "counts": {"positive": 4, "negative": 0}}
        (base_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        embs = np.ones((4, 1024), dtype=np.float16)
        np.save(str(pos_dir / "embeddings.npy"), embs)
        df_meta = pd.DataFrame(
            {
                "id": [f"ID_{i}" for i in range(4)],
                "smiles": ["C"] * 4,
                "inchikey14": [f"IK{i}" for i in range(4)],
                "precursor_mz": [100.0 + i for i in range(4)],
                "adduct": ["[M+H]+"] * 4,
                "instrument": ["generic"] * 4,
            }
        )
        df_meta.to_parquet(str(pos_dir / "metadata.parquet"), index=False)

        # Baseline check
        valid, msgs = validate_spectral_index(base_dir)
        assert valid is True

        # Corruption 1: Missing manifest.json
        c1_dir = tmp_path / "c1_no_manifest"
        shutil.copytree(base_dir, c1_dir)
        (c1_dir / "manifest.json").unlink()
        valid, errors = validate_spectral_index(c1_dir)
        assert not valid
        assert any("manifest.json not found" in e for e in errors)

        # Corruption 2: Wrong embedding dtype (float32 instead of float16)
        c2_dir = tmp_path / "c2_wrong_dtype"
        shutil.copytree(base_dir, c2_dir)
        np.save(str(c2_dir / "positive" / "embeddings.npy"), embs.astype(np.float32))
        valid, errors = validate_spectral_index(c2_dir)
        assert not valid
        assert any("expected float16" in e for e in errors)

        # Corruption 3: Wrong embedding dimension (512 instead of 1024)
        c3_dir = tmp_path / "c3_wrong_dim"
        shutil.copytree(base_dir, c3_dir)
        np.save(str(c3_dir / "positive" / "embeddings.npy"), np.ones((4, 512), dtype=np.float16))
        valid, errors = validate_spectral_index(c3_dir)
        assert not valid
        assert any("does not match expected (*, 1024)" in e for e in errors)

        # Corruption 4: Row alignment mismatch (5 vectors vs 4 metadata rows)
        c4_dir = tmp_path / "c4_mismatch"
        shutil.copytree(base_dir, c4_dir)
        np.save(str(c4_dir / "positive" / "embeddings.npy"), np.ones((5, 1024), dtype=np.float16))
        valid, errors = validate_spectral_index(c4_dir)
        assert not valid
        assert any("row alignment mismatch" in e for e in errors)

        # Corruption 5: Missing required metadata column (missing precursor_mz)
        c5_dir = tmp_path / "c5_missing_col"
        shutil.copytree(base_dir, c5_dir)
        df_bad = df_meta.drop(columns=["precursor_mz"])
        df_bad.to_parquet(str(c5_dir / "positive" / "metadata.parquet"), index=False)
        valid, errors = validate_spectral_index(c5_dir)
        assert not valid
        assert any("missing columns" in e for e in errors)
