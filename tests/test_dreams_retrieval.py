import numpy as np
import pytest
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever, RetrievedCandidate


def test_instrument_stratified_threshold():
    retriever = CalibratedDreaMSRetriever()
    # timsTOF-to-timsTOF threshold is 0.88, while cross-instrument is 0.82 (Orbitrap) and 0.80 (QTOF)
    assert retriever.get_calibration_threshold("timsTOF", "timsTOF") == 0.88
    assert retriever.get_calibration_threshold("timsTOF", "Orbitrap") == 0.82
    assert retriever.get_calibration_threshold("timsTOF", "QTOF") == 0.80
    # Generic fallback for unknown instruments
    assert retriever.get_calibration_threshold("unknown", "unknown") == 0.85
    assert retriever.get_calibration_threshold("timsTOF", "UnknownMS") == 0.85


def test_fp16_rescoring_and_pinning():
    retriever = CalibratedDreaMSRetriever()
    np.random.seed(42)
    query_emb = np.random.randn(1024).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    # Mock candidate matching with cosine ~ 1.0 (>= 0.88 on timsTOF)
    candidate_emb = query_emb.copy()
    cands, is_locked = retriever.evaluate_candidates(
        query_emb=query_emb,
        raw_candidates=[("C1=CC=CC=C1", "INCHIKEY123456", candidate_emb, "timsTOF")],
        query_instrument="timsTOF",
    )
    assert len(cands) == 1
    assert is_locked is True
    assert cands[0].score > 0.99
    assert cands[0].smiles == "C1=CC=CC=C1"
    assert cands[0].inchikey14 == "INCHIKEY123456"
    assert cands[0].instrument == "timsTOF"
    assert cands[0].source_tier == "track1_dreams"


def test_multiple_candidates_sorting_by_score():
    retriever = CalibratedDreaMSRetriever()
    query_emb = np.zeros(1024, dtype=np.float32)
    query_emb[0] = 1.0

    # Cand A: cosine similarity 1.0
    emb_a = np.zeros(1024, dtype=np.float32)
    emb_a[0] = 1.0

    # Cand B: cosine similarity ~ 0.8
    emb_b = np.zeros(1024, dtype=np.float32)
    emb_b[0] = 0.8
    emb_b[1] = 0.6

    # Cand C: cosine similarity 0.0
    emb_c = np.zeros(1024, dtype=np.float32)
    emb_c[1] = 1.0

    raw_cands = [
        ("SMILES_C", "IK14_C", emb_c, "Orbitrap"),
        ("SMILES_A", "IK14_A", emb_a, "timsTOF"),
        ("SMILES_B", "IK14_B", emb_b, "QTOF"),
    ]

    cands, is_locked = retriever.evaluate_candidates(
        query_emb=query_emb,
        raw_candidates=raw_cands,
        query_instrument="timsTOF",
    )
    assert len(cands) == 3
    assert [c.smiles for c in cands] == ["SMILES_A", "SMILES_B", "SMILES_C"]
    assert cands[0].score > cands[1].score > cands[2].score
    assert is_locked is True


def test_candidates_below_threshold_not_locked():
    retriever = CalibratedDreaMSRetriever()
    query_emb = np.zeros(1024, dtype=np.float32)
    query_emb[0] = 1.0

    # Cand has cosine similarity ~ 0.70, which is below 0.88 (timsTOF)
    emb = np.zeros(1024, dtype=np.float32)
    emb[0] = 0.70
    emb[1] = np.sqrt(1 - 0.70**2)

    cands, is_locked = retriever.evaluate_candidates(
        query_emb=query_emb,
        raw_candidates=[("C1=CC=CC=C1", "INCHIKEY123456", emb, "timsTOF")],
        query_instrument="timsTOF",
    )
    assert len(cands) == 1
    assert is_locked is False
    assert 0.69 < cands[0].score < 0.71


def test_empty_candidates_handling():
    retriever = CalibratedDreaMSRetriever()
    query_emb = np.ones(1024, dtype=np.float32)
    cands, is_locked = retriever.evaluate_candidates(
        query_emb=query_emb,
        raw_candidates=[],
        query_instrument="timsTOF",
    )
    assert cands == []
    assert is_locked is False


def test_fp16_precision_behavior_and_safeguards():
    retriever = CalibratedDreaMSRetriever()
    query_emb = np.random.randn(1024).astype(np.float32)

    zero_emb = np.zeros(1024, dtype=np.float32)
    cands, is_locked = retriever.evaluate_candidates(
        query_emb=query_emb,
        raw_candidates=[("ZERO_VEC", "IK14ZERO000000", zero_emb, "generic")],
        query_instrument="generic",
    )
    assert len(cands) == 1
    assert not np.isnan(cands[0].score)
    assert not np.isinf(cands[0].score)
    assert is_locked is False
