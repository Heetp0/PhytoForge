"""
tests/test_spectral_indexer.py
Comprehensive unit and integration test suite for memory-mapped reference
spectral vector indexer and CalibratedDreaMSRetriever dependency injection
(Milestone M5, R2 & R4 DI).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.data.spectral_indexer import (
    SpectralHit,
    SpectralIndex,
    SpectralIndexBuilder,
    normalize_polarity,
)
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever, RetrievedCandidate


# ---------------------------------------------------------------------------
# Synthetic Dataset Generation Helper
# ---------------------------------------------------------------------------

def create_synthetic_spectra_dataset(
    n_positive: int = 10,
    n_negative: int = 6,
    dim: int = 1024,
    random_state: int = 42,
) -> pd.DataFrame:
    """Creates a deterministic synthetic spectra DataFrame with known properties."""
    rng = np.random.default_rng(random_state)
    records = []

    # Positive mode spectra
    base_mzs_pos = [150.05, 200.10, 250.15, 300.20, 350.25, 400.30, 450.35, 500.40, 550.45, 600.50]
    instruments = ["timsTOF", "Orbitrap", "QTOF", "generic"]

    for i in range(n_positive):
        mz = base_mzs_pos[i % len(base_mzs_pos)] + (i * 0.001)
        raw_vec = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(raw_vec)
        unit_vec = (raw_vec / (norm + 1e-12)).astype(np.float32)

        records.append(
            {
                "id": f"POS_{i:04d}",
                "smiles": f"C{i+5}H{2*i+6}O2",
                "inchikey14": f"POSIK14{i:07d}",
                "precursor_mz": mz,
                "adduct": "[M+H]+",
                "instrument": instruments[i % len(instruments)],
                "polarity": "positive",
                "embedding": unit_vec,
            }
        )

    # Negative mode spectra
    base_mzs_neg = [148.03, 198.08, 248.13, 298.18, 348.23, 398.28]
    for i in range(n_negative):
        mz = base_mzs_neg[i % len(base_mzs_neg)] + (i * 0.001)
        raw_vec = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(raw_vec)
        unit_vec = (raw_vec / (norm + 1e-12)).astype(np.float32)

        records.append(
            {
                "id": f"NEG_{i:04d}",
                "smiles": f"C{i+4}H{2*i+4}O3",
                "inchikey14": f"NEGIK14{i:07d}",
                "precursor_mz": mz,
                "adduct": "[M-H]-",
                "instrument": instruments[i % len(instruments)],
                "polarity": "negative",
                "embedding": unit_vec,
            }
        )

    return pd.DataFrame(records)


# ===========================================================================
# 1. SpectralIndexBuilder Tests
# ===========================================================================

class TestSpectralIndexBuilder:
    """Test building, partitioning, sorting, and serialization of spectral indexes."""

    def test_build_from_dataframe(self, tmp_path: Path):
        df = create_synthetic_spectra_dataset(n_positive=8, n_negative=4)
        out_dir = tmp_path / "spectral_index"

        result_path = SpectralIndexBuilder.build_index(df, output_dir=out_dir, dim=1024)
        assert result_path == out_dir
        assert (out_dir / "manifest.json").exists()

        # Check partitions
        for mode in ("positive", "negative"):
            part_dir = out_dir / mode
            assert (part_dir / "embeddings.npy").exists()
            assert (part_dir / "metadata.parquet").exists()

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["version"] == "1.0.0"
        assert manifest["dim"] == 1024
        assert manifest["counts"]["positive"] == 8
        assert manifest["counts"]["negative"] == 4
        assert "created_at" in manifest

    def test_build_from_numpy_arrays_and_metadata(self, tmp_path: Path):
        out_dir = tmp_path / "from_arrays"
        rng = np.random.default_rng(77)
        embs = rng.standard_normal((6, 1024)).astype(np.float32)
        meta = pd.DataFrame(
            {
                "id": [f"S_{i}" for i in range(6)],
                "smiles": ["CC(=O)O"] * 6,
                "precursor_mz": [100.0, 150.0, 200.0, 105.0, 155.0, 205.0],
                "adduct": ["[M+H]+"] * 3 + ["[M-H]-"] * 3,
                "instrument": ["timsTOF"] * 6,
            }
        )

        SpectralIndexBuilder.build_index(
            embeddings=embs,
            metadata=meta,
            output_dir=out_dir,
            dim=1024,
        )

        assert (out_dir / "positive" / "embeddings.npy").exists()
        assert (out_dir / "negative" / "embeddings.npy").exists()

    def test_build_from_parquet_and_csv_files(self, tmp_path: Path):
        df = create_synthetic_spectra_dataset(n_positive=5, n_negative=3)
        pq_path = tmp_path / "spectra_input.parquet"
        df.to_parquet(pq_path, index=False)

        out_dir = tmp_path / "from_pq_file"
        SpectralIndexBuilder.build_index(spectra=pq_path, output_dir=out_dir)
        assert (out_dir / "manifest.json").exists()

    def test_precursor_sorting_guarantee(self, tmp_path: Path):
        """Verify precursor_mz is strictly monotonically sorted ascending."""
        rng = np.random.default_rng(88)
        # Unsorted precursor masses
        unsorted_mzs = [500.0, 200.0, 350.0, 150.0, 600.0]
        df = pd.DataFrame(
            {
                "id": [f"S_{i}" for i in range(5)],
                "smiles": ["C6H6"] * 5,
                "precursor_mz": unsorted_mzs,
                "adduct": ["[M+H]+"] * 5,
                "polarity": ["positive"] * 5,
                "embedding": list(rng.standard_normal((5, 1024)).astype(np.float32)),
            }
        )

        out_dir = tmp_path / "sorted_index"
        SpectralIndexBuilder.build_index(df, output_dir=out_dir)

        df_meta = pd.read_parquet(out_dir / "positive" / "metadata.parquet")
        masses = df_meta["precursor_mz"].to_numpy()
        assert np.all(np.diff(masses) >= 0), f"Precursor masses were not sorted: {masses}"
        assert list(masses) == sorted(unsorted_mzs)

    def test_embedding_normalization_l2(self, tmp_path: Path):
        """Verify embeddings are L2 normalized to unit norm in float16."""
        rng = np.random.default_rng(99)
        raw_embs = rng.uniform(5.0, 50.0, size=(4, 1024)).astype(np.float32)
        df = pd.DataFrame(
            {
                "id": [f"ID_{i}" for i in range(4)],
                "smiles": ["C6H6"] * 4,
                "precursor_mz": [100.0, 120.0, 140.0, 160.0],
                "adduct": ["[M+H]+"] * 4,
                "embedding": list(raw_embs),
            }
        )

        out_dir = tmp_path / "normalized_index"
        SpectralIndexBuilder.build_index(df, output_dir=out_dir)

        saved_embs = np.load(out_dir / "positive" / "embeddings.npy")
        assert saved_embs.dtype == np.float16
        norms = np.linalg.norm(saved_embs.astype(np.float32), axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-3)

    def test_empty_partition_handling(self, tmp_path: Path):
        """When input has only positive spectra, negative partition is safely initialized empty."""
        df = create_synthetic_spectra_dataset(n_positive=5, n_negative=0)
        out_dir = tmp_path / "positive_only"
        SpectralIndexBuilder.build_index(df, output_dir=out_dir)

        with SpectralIndex(out_dir) as idx:
            assert idx.partitions["positive"]["count"] == 5
            assert idx.partitions["negative"]["count"] == 0
            assert idx.partitions["negative"]["embeddings"].shape == (0, 1024)
            assert len(idx.partitions["negative"]["metadata"]) == 0


# ===========================================================================
# 2. SpectralIndex Tests
# ===========================================================================

class TestSpectralIndex:
    """Test zero-copy memory-mapping, row alignment, mass windowing, and cosine search."""

    @pytest.fixture
    def built_index(self, tmp_path: Path) -> Path:
        out_dir = tmp_path / "fixture_index"
        df = create_synthetic_spectra_dataset(n_positive=10, n_negative=6)
        SpectralIndexBuilder.build_index(df, output_dir=out_dir)
        return out_dir

    def test_zero_copy_memmap_properties(self, built_index: Path):
        with SpectralIndex(built_index) as idx:
            embs = idx.get_embeddings("positive")
            assert isinstance(embs, np.memmap), "Embeddings must be an instance of np.memmap"
            assert embs.mode == "r"
            assert embs.dtype == np.float16
            assert embs.shape == (10, 1024)

            # Assert write protection
            with pytest.raises(ValueError):
                embs[0, 0] = 99.0

    def test_one_to_one_row_alignment(self, built_index: Path):
        with SpectralIndex(built_index) as idx:
            for mode, expected_cnt in [("positive", 10), ("negative", 6)]:
                embs = idx.get_embeddings(mode)
                meta = idx.get_metadata(mode)
                assert len(embs) == len(meta) == expected_cnt
                required_cols = {"id", "smiles", "inchikey14", "precursor_mz", "adduct", "instrument"}
                assert required_cols.issubset(set(meta.columns))

    def test_mass_window_filtering_10ppm(self, tmp_path: Path):
        """Verify candidate slice is strictly pre-filtered within +- 10 ppm."""
        out_dir = tmp_path / "mass_window_index"
        rng = np.random.default_rng(101)
        # Precursor masses:
        # Query mz = 200.0.
        # 10 ppm tolerance: 200.0 * 10e-6 = 0.0020 Da.
        # In-window: 200.000, 200.001 (0.001 Da diff = 5 ppm).
        # Out-of-window: 200.010 (0.010 Da diff = 50 ppm), 199.980 (0.020 Da diff = 100 ppm), 300.000.
        mzs = [199.980, 200.000, 200.001, 200.010, 300.000]
        embs = rng.standard_normal((len(mzs), 1024)).astype(np.float32)

        df = pd.DataFrame(
            {
                "id": [f"ID_{i}" for i in range(len(mzs))],
                "smiles": ["C6H6"] * len(mzs),
                "inchikey14": [f"IK14_{i:06d}" for i in range(len(mzs))],
                "precursor_mz": mzs,
                "adduct": ["[M+H]+"] * len(mzs),
                "instrument": ["timsTOF"] * len(mzs),
                "embedding": list(embs),
            }
        )
        SpectralIndexBuilder.build_index(df, output_dir=out_dir)

        with SpectralIndex(out_dir) as idx:
            # Query at 200.0 with 10.0 ppm
            query_emb = embs[1]  # Vector matching ID_1 (200.000)
            hits = idx.search(query_emb, precursor_mz=200.0, ppm_tolerance=10.0)

            retrieved_ids = {h.id for h in hits}
            # Only ID_1 (200.000) and ID_2 (200.001) should be within +- 10 ppm
            assert retrieved_ids == {"ID_1", "ID_2"}
            assert "ID_0" not in retrieved_ids
            assert "ID_3" not in retrieved_ids
            assert "ID_4" not in retrieved_ids

    def test_cosine_similarity_and_ranking(self, built_index: Path):
        with SpectralIndex(built_index) as idx:
            meta = idx.get_metadata("positive")
            target_row = meta.iloc[3]
            target_mz = float(target_row["precursor_mz"])
            target_emb = idx.get_embeddings("positive")[3].astype(np.float32)

            hits = idx.search(
                query_emb=target_emb,
                precursor_mz=target_mz,
                polarity="positive",
                ppm_tolerance=10.0,
                top_k=5,
            )

            assert len(hits) >= 1
            top_hit = hits[0]
            assert top_hit.id == target_row["id"]
            assert top_hit.smiles == target_row["smiles"]
            assert top_hit.inchikey14 == target_row["inchikey14"]
            assert top_hit.rank == 1
            assert top_hit.cosine_score == pytest.approx(1.0, abs=1e-3)

            # Verify descending score sort
            scores = [h.cosine_score for h in hits]
            assert scores == sorted(scores, reverse=True)

    def test_spectral_hit_to_candidate_conversion(self):
        hit = SpectralHit(
            id="SPEC_001",
            smiles="CC(=O)O",
            inchikey14="QTBSBXVTEAMEQO",
            precursor_mz=61.0284,
            adduct="[M+H]+",
            instrument="timsTOF",
            cosine_score=0.92,
            rank=1,
        )
        cand = hit.to_candidate()
        assert isinstance(cand, RetrievedCandidate)
        assert cand.smiles == "CC(=O)O"
        assert cand.inchikey14 == "QTBSBXVTEAMEQO"
        assert cand.score == 0.92
        assert cand.instrument == "timsTOF"
        assert cand.source_tier == "track1_dreams"

    def test_edge_cases(self, built_index: Path):
        with SpectralIndex(built_index) as idx:
            dummy_vec = np.ones(1024, dtype=np.float32)

            # 1. precursor_mz <= 0 returns empty
            assert idx.search(dummy_vec, precursor_mz=0.0) == []
            assert idx.search(dummy_vec, precursor_mz=-10.0) == []

            # 2. No matching candidates in mass window
            assert idx.search(dummy_vec, precursor_mz=99999.0) == []

            # 3. Query vector with zero norm
            zero_vec = np.zeros(1024, dtype=np.float32)
            res = idx.search(zero_vec, precursor_mz=200.10)
            if res:
                assert all(h.cosine_score == 0.0 for h in res)

    def test_polarity_normalization(self):
        assert normalize_polarity("positive") == "positive"
        assert normalize_polarity("pos") == "positive"
        assert normalize_polarity("+") == "positive"
        assert normalize_polarity(adduct="[M+H]+") == "positive"

        assert normalize_polarity("negative") == "negative"
        assert normalize_polarity("neg") == "negative"
        assert normalize_polarity("-") == "negative"
        assert normalize_polarity(adduct="[M-H]-") == "negative"

    def test_windows_cleanup_and_context_manager(self, tmp_path: Path):
        """Verify close() releases mmap handle so directory can be removed cleanly on Windows."""
        temp_dir = tmp_path / "cleanup_test"
        df = create_synthetic_spectra_dataset(n_positive=4, n_negative=2)
        SpectralIndexBuilder.build_index(df, output_dir=temp_dir)

        idx = SpectralIndex(temp_dir)
        _ = idx.search(np.ones(1024), precursor_mz=150.05)
        # Explicit close releases Windows file lock
        idx.close()

        # Should delete without WinError 32 PermissionError
        shutil.rmtree(temp_dir)
        assert not temp_dir.exists()


# ===========================================================================
# 3. CalibratedDreaMSRetriever Integration Tests
# ===========================================================================

class TestCalibratedDreaMSRetrieverIntegration:
    """Test CalibratedDreaMSRetriever dependency injection and instrument calibration."""

    @pytest.fixture
    def built_index(self, tmp_path: Path) -> Path:
        out_dir = tmp_path / "retriever_test_index"
        df = create_synthetic_spectra_dataset(n_positive=10, n_negative=6)
        SpectralIndexBuilder.build_index(df, output_dir=out_dir)
        return out_dir

    def test_retriever_with_index_path(self, built_index: Path):
        retriever = CalibratedDreaMSRetriever(index=built_index)
        assert retriever.has_index is True
        assert retriever.index is not None

        # Clean up
        if hasattr(retriever.index, "close"):
            retriever.index.close()

    def test_retriever_retrieve_method_and_locking(self, built_index: Path):
        with SpectralIndex(built_index) as idx:
            retriever = CalibratedDreaMSRetriever(index=idx)

            # Query identical to candidate 0 (instrument timsTOF)
            target_emb = idx.get_embeddings("positive")[0].astype(np.float32)
            target_mz = float(idx.get_metadata("positive").iloc[0]["precursor_mz"])

            candidates, is_locked = retriever.retrieve(
                query_emb=target_emb,
                precursor_mz=target_mz,
                polarity="positive",
                query_instrument="timsTOF",
            )

            assert len(candidates) >= 1
            top = candidates[0]
            assert top.score == pytest.approx(1.0, abs=1e-3)
            # Since score 1.0 >= timsTOF->timsTOF threshold 0.88, is_locked must be True
            assert is_locked is True

    def test_instrument_stratified_thresholds(self, built_index: Path):
        retriever = CalibratedDreaMSRetriever()
        assert retriever.get_calibration_threshold("timsTOF", "timsTOF") == 0.88
        assert retriever.get_calibration_threshold("timsTOF", "Orbitrap") == 0.82
        assert retriever.get_calibration_threshold("timsTOF", "QTOF") == 0.80
        assert retriever.get_calibration_threshold("generic", "generic") == 0.85

    def test_backward_compatibility_with_raw_candidates(self):
        """Verify legacy evaluation executes unchanged when raw_candidates is passed."""
        retriever = CalibratedDreaMSRetriever()
        assert retriever.has_index is False

        q_vec = np.zeros(1024, dtype=np.float32)
        q_vec[0] = 1.0
        c1_vec = np.zeros(1024, dtype=np.float32)
        c1_vec[0] = 1.0  # Perfect match
        c2_vec = np.zeros(1024, dtype=np.float32)
        c2_vec[1] = 1.0  # Orthogonal

        raw_candidates = [
            ("SMI_1", "IK14_1", c1_vec, "timsTOF"),
            ("SMI_2", "IK14_2", c2_vec, "Orbitrap"),
        ]

        results, is_locked = retriever.evaluate_candidates(
            query_emb=q_vec,
            raw_candidates=raw_candidates,
            query_instrument="timsTOF",
        )

        assert len(results) == 2
        assert results[0].smiles == "SMI_1"
        assert results[0].score == pytest.approx(1.0, abs=1e-3)
        assert results[1].smiles == "SMI_2"
        assert results[1].score == pytest.approx(0.0, abs=1e-3)
        assert is_locked is True

    def test_evaluate_candidates_delegates_to_retrieve(self, built_index: Path):
        with SpectralIndex(built_index) as idx:
            retriever = CalibratedDreaMSRetriever(index=idx)
            target_emb = idx.get_embeddings("positive")[0].astype(np.float32)
            target_mz = float(idx.get_metadata("positive").iloc[0]["precursor_mz"])

            # Pass raw_candidates=None to trigger delegation
            results, is_locked = retriever.evaluate_candidates(
                query_emb=target_emb,
                raw_candidates=None,
                precursor_mz=target_mz,
                polarity="positive",
            )

            assert len(results) >= 1
            assert results[0].score == pytest.approx(1.0, abs=1e-3)
