import numpy as np
import pytest
from src.reranking.meta_ranker import CandidateFeatureVector, GBDTMetaRanker


def test_feature_vector_dimension_ge_30():
    ranker = GBDTMetaRanker()
    vec = ranker.extract_feature_vector(
        dreams_cosine=0.85,
        mist_tanimoto=0.72,
        ppm_error=2.5,
        abs_da_error=0.0012,
        fragment_match_ratio=0.6,
        np_score=1.45,
        entropy_similarity=0.78,
        cross_track_agreement_count=2,
        formula_rank=1,
        source_track="track1_dreams",
    )
    assert len(vec) >= 30
    assert len(vec) == 32
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_reranking_sort_order():
    ranker = GBDTMetaRanker()
    cands = [
        {"id": "cand_low", "features": np.zeros(32, dtype=np.float32)},
        {"id": "cand_mid", "features": np.full(32, 0.5, dtype=np.float32)},
        {"id": "cand_high", "features": np.ones(32, dtype=np.float32)},
    ]
    scored = ranker.score_candidates(cands)
    assert len(scored) == 3
    assert scored[0]["id"] == "cand_high"
    assert scored[1]["id"] == "cand_mid"
    assert scored[2]["id"] == "cand_low"
    assert scored[0]["meta_score"] > scored[1]["meta_score"] > scored[2]["meta_score"]


def test_one_hot_track_encodings():
    ranker = GBDTMetaRanker()
    tracks = ["track1_dreams", "track2_db", "track3_denovo", "track_network"]
    expected_indices = [12, 13, 14, 15]

    for track, exp_idx in zip(tracks, expected_indices):
        vec = ranker.extract_feature_vector(
            dreams_cosine=0.5,
            mist_tanimoto=0.5,
            ppm_error=0.0,
            abs_da_error=0.0,
            fragment_match_ratio=0.5,
            np_score=1.0,
            entropy_similarity=0.5,
            cross_track_agreement_count=1,
            formula_rank=1,
            source_track=track,
        )
        assert vec[exp_idx] == 1.0
        # All other track indices in 12..15 should be 0.0
        for other_idx in expected_indices:
            if other_idx != exp_idx:
                assert vec[other_idx] == 0.0

    # Unknown track
    vec_unknown = ranker.extract_feature_vector(
        dreams_cosine=0.5,
        mist_tanimoto=0.5,
        ppm_error=0.0,
        abs_da_error=0.0,
        fragment_match_ratio=0.5,
        np_score=1.0,
        entropy_similarity=0.5,
        cross_track_agreement_count=1,
        formula_rank=1,
        source_track="unknown_track",
    )
    for idx in expected_indices:
        assert vec_unknown[idx] == 0.0


def test_custom_weights_handling():
    weights = np.zeros(32, dtype=np.float32)
    weights[0] = 10.0  # only dreams_cosine matters
    ranker = GBDTMetaRanker(weights=weights)

    cand_a = {"id": "A", "features": np.zeros(32, dtype=np.float32)}
    cand_a["features"][0] = 0.9
    cand_b = {"id": "B", "features": np.ones(32, dtype=np.float32)}
    cand_b["features"][0] = 0.1

    scored = ranker.score_candidates([cand_b, cand_a])
    assert scored[0]["id"] == "A"
    assert pytest.approx(scored[0]["meta_score"], 1e-4) == 9.0
    assert pytest.approx(scored[1]["meta_score"], 1e-4) == 1.0


def test_score_computation_numerical_stability():
    ranker = GBDTMetaRanker()
    # Extreme inputs: huge ppm error, negative ppm, 0 formula rank
    vec = ranker.extract_feature_vector(
        dreams_cosine=-1.0,
        mist_tanimoto=0.0,
        ppm_error=1e6,
        abs_da_error=1e4,
        fragment_match_ratio=0.0,
        np_score=-5.0,
        entropy_similarity=0.0,
        cross_track_agreement_count=0,
        formula_rank=0,  # Should not cause ZeroDivisionError
        source_track="",
    )
    assert len(vec) == 32
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    # vec[11] = 1.0 / max(1, 0) == 1.0
    assert vec[11] == 1.0
    # Mass accuracy reciprocal transforms
    assert 0.0 <= vec[6] <= 1.0
    assert 0.0 <= vec[7] <= 1.0

    # CandidateFeatureVector dataclass instantiation
    cfv = CandidateFeatureVector(features=vec)
    assert np.array_equal(cfv.features, vec)
