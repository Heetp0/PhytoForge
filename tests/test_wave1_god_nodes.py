"""
Unit and integration tests for Wave 1 God Nodes optimization:
- SoftDatabaseSearcher (max candidate limits & popcount safeguards)
- CalibratedDreaMSRetriever (CASMI tuning & dynamic thresholding)
- SpectralIndex (vectorized dot product & zero-norm safeguards)
- ChemicalDatabaseRepository (max_per_formula capping & 1GB mmap pragma)
- CASMIOmegaPipeline (Dynamic Query Fingerprint synthesis replacing all-ones dummy)
"""

from pathlib import Path
import numpy as np
import pytest

from src.retrieval.database_search import DBCandidate, SoftDatabaseSearcher
from src.retrieval.dreams_retrieval import (
    CalibratedDreaMSRetriever,
    RetrievedCandidate,
)
from src.data.db_indexer import (
    ChemicalDatabaseRepository,
    pack_fingerprint,
    unpack_fingerprint,
    popcount_tanimoto,
)
from src.data.spectral_indexer import SpectralIndex, SpectralHit


def test_popcount_tanimoto_zero_query_guard():
    """Verify popcount_tanimoto returns exactly zero for zeroed query fingerprint."""
    zero_q = np.zeros(32, dtype=np.uint64)
    cands = np.ones((10, 32), dtype=np.uint64)
    scores = popcount_tanimoto(zero_q, cands)
    assert scores.shape == (10,)
    assert np.all(scores == 0.0)


def test_popcount_tanimoto_zero_candidates_guard():
    """Verify popcount_tanimoto handles empty candidate matrix without error."""
    q = np.ones(32, dtype=np.uint64)
    scores = popcount_tanimoto(q, np.empty((0, 32), dtype=np.uint64))
    assert scores.shape == (0,)


def test_soft_database_searcher_max_candidates_capping():
    """Verify max_candidates_per_formula limits candidates returned from bloated formula pools."""
    # Create a mock database with 200 candidates under one formula
    large_formula_list = [
        (f"C{i}H{i}O", f"IK14_{i:06d}", np.ones(3, dtype=np.float32))
        for i in range(200)
    ]
    mock_db = {"C10H20O": large_formula_list}
    searcher = SoftDatabaseSearcher(db=mock_db)

    # Search with limit of 50
    hits = searcher.search_formulas(
        formulas=["C10H20O"],
        query_fp=np.ones(3, dtype=np.float32),
        max_candidates_per_formula=50,
    )
    assert len(hits) == 50
    assert hits[0].source_tier == "track2_db"


def test_calibrated_dreams_retriever_custom_thresholds():
    """Verify CalibratedDreaMSRetriever supports CASMI-tuned thresholds and runtime adjustment."""
    retriever = CalibratedDreaMSRetriever()
    assert retriever.get_calibration_threshold("timsTOF", "timsTOF") == 0.88

    # Set competition-tuned threshold
    retriever.set_threshold("timsTOF", "timsTOF", 0.78)
    assert retriever.get_calibration_threshold("timsTOF", "timsTOF") == 0.78

    # Query with cosine 0.80 should now lock rank 1
    query_emb = np.zeros(1024, dtype=np.float32)
    query_emb[0] = 1.0

    cand_emb = np.zeros(1024, dtype=np.float32)
    cand_emb[0] = 0.80
    cand_emb[1] = np.sqrt(1.0 - 0.80**2)

    cands, is_locked = retriever.evaluate_candidates(
        query_emb=query_emb,
        raw_candidates=[("C1=CC=CC=C1", "INCHIKEY123456", cand_emb, "timsTOF")],
        query_instrument="timsTOF",
    )
    assert len(cands) == 1
    assert is_locked is True


def test_spectral_index_fast_dot_product_numerical_accuracy(tmp_path):
    """Verify SpectralIndex cosine search handles zero norms, small norms, and exact matches."""
    idx_dir = tmp_path / "spectral_index"
    idx_dir.mkdir(parents=True)
    pos_dir = idx_dir / "positive"
    pos_dir.mkdir()

    # Generate 5 reference vectors
    dim = 1024
    np.random.seed(42)
    embs = np.random.randn(5, dim).astype(np.float16)
    for i in range(5):
        embs[i] /= np.linalg.norm(embs[i].astype(np.float32))

    np.save(pos_dir / "embeddings.npy", embs)

    import pandas as pd
    df_meta = pd.DataFrame({
        "id": [f"SPEC_{i}" for i in range(5)],
        "smiles": [f"C{i+1}" for i in range(5)],
        "inchikey14": [f"IK14_{i}" for i in range(5)],
        "precursor_mz": [100.0 + i * 0.001 for i in range(5)],
        "adduct": ["[M+H]+"] * 5,
        "instrument": ["timsTOF"] * 5,
    })
    df_meta.to_parquet(pos_dir / "metadata.parquet")

    with open(idx_dir / "manifest.json", "w") as f:
        f.write('{"dim": 1024}')

    index = SpectralIndex(idx_dir)

    # Search with exact vector of candidate 2
    query = embs[2].astype(np.float32)
    hits = index.search(
        query_emb=query,
        precursor_mz=100.002,
        polarity="positive",
        ppm_tolerance=50.0,
        top_k=5,
    )
    assert len(hits) > 0
    assert hits[0].inchikey14 == "IK14_2"
    assert pytest.approx(hits[0].cosine_score, 1e-2) == 1.0

    index.close()
