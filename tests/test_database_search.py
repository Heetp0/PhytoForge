import numpy as np
import pytest
from src.retrieval.database_search import DBCandidate, SoftDatabaseSearcher


def test_database_search_with_formula_union():
    # Mock database with two formulas
    mock_db = {
        "C15H10O5": [("c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1", "IK14_A", np.array([1, 0, 1]))],
        "C15H10O6": [("c1cc(O)c2c(=O)cc(-c3ccc(O)cc3)oc2c1", "IK14_B", np.array([1, 1, 1]))],
    }
    fallback = [("C1CCCCC1", "IK14_FALLBACK", np.array([0, 0, 0]))]

    searcher = SoftDatabaseSearcher(db=mock_db, fallback_scaffolds=fallback)
    query_fp = np.array([1, 1, 1])

    hits = searcher.search_formulas(["C15H10O5", "C15H10O6"], query_fp=query_fp)
    assert len(hits) == 2
    assert hits[0].inchikey14 == "IK14_B"  # Exact fingerprint match ranked first
    assert hits[0].source_tier == "track2_db"
    assert pytest.approx(hits[0].tanimoto_score, 1e-5) == 1.0


def test_zero_hit_fallback_activation():
    searcher = SoftDatabaseSearcher(
        db={},
        fallback_scaffolds=[("C1CCCCC1", "IK14_FALLBACK", np.array([0, 0, 0]))],
    )
    hits = searcher.search_formulas(["C99H99O99"], query_fp=np.array([1, 0, 0]))
    assert len(hits) == 1
    assert hits[0].inchikey14 == "IK14_FALLBACK"
    assert hits[0].source_tier == "track2_fallback"
    assert hits[0].tanimoto_score == 0.01


def test_deduplication_by_inchikey14_across_formulas():
    mock_db = {
        "C6H6": [("c1ccccc1", "IK14_BENZENE", np.array([1, 0, 0]))],
        "C6H6_ALT": [("C1=CC=CC=C1", "IK14_BENZENE", np.array([1, 0, 0]))],
    }
    searcher = SoftDatabaseSearcher(db=mock_db, fallback_scaffolds=[])
    query_fp = np.array([1, 0, 0])

    hits = searcher.search_formulas(["C6H6", "C6H6_ALT"], query_fp=query_fp)
    assert len(hits) == 1
    assert hits[0].inchikey14 == "IK14_BENZENE"


def test_empty_fallback_list_safety():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])
    hits = searcher.search_formulas(["C10H10"], query_fp=np.array([1, 1, 0]))
    assert hits == []


def test_vectorized_batch_tanimoto_vs_pairwise_equivalence():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])

    rng = np.random.default_rng(seed=42)
    query_fp = rng.integers(0, 2, size=128)
    db_fps = rng.integers(0, 2, size=(50, 128))

    batch_scores = searcher.batch_tanimoto(query_fp, db_fps)
    assert batch_scores.shape == (50,)

    pairwise_scores = np.array([searcher.tanimoto(query_fp, fp) for fp in db_fps], dtype=np.float32)
    assert np.allclose(batch_scores, pairwise_scores, atol=1e-5)


def test_batch_tanimoto_edge_cases_and_safeguards():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])

    empty_res = searcher.batch_tanimoto(np.array([1, 0]), np.empty((0, 2)))
    assert empty_res.size == 0

    zero_query = np.array([0, 0, 0])
    zero_db = np.array([[0, 0, 0], [1, 0, 0]])
    scores = searcher.batch_tanimoto(zero_query, zero_db)
    assert scores.shape == (2,)
    assert scores[0] == 0.0
    assert scores[1] == 0.0
