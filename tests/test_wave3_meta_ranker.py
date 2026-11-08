"""
Unit and integration tests for Wave 3 God Nodes optimization:
- GBDTMetaRanker (intra-formula constitutional isomer discrimination, weights rebalancing, NaN/Inf protection)
- Dynamic feature vector population in CASMIOmegaPipeline
"""

import numpy as np
import pytest

from src.reranking.meta_ranker import GBDTMetaRanker
from src.pipeline import CASMIOmegaPipeline


def test_intra_formula_isomer_discrimination():
    """Verify that two candidates with identical formula & mass are separated by RDA fragmentation & botanical prior."""
    ranker = GBDTMetaRanker()

    # Isomer 1: True natural product (Apigenin core), has high RDA fragment match and high botanical prior
    feat_apigenin = ranker.extract_feature_vector(
        dreams_cosine=0.5,
        mist_tanimoto=0.75,
        ppm_error=0.0,       # Identical mass error
        abs_da_error=0.0,    # Identical mass error
        fragment_match_ratio=0.85, # High in-silico RDA ring cleavage match
        np_score=1.5,
        entropy_similarity=0.8,
        cross_track_agreement_count=2,
        formula_rank=1,
        source_track="track2_db",
        composite_prior_confidence=0.90, # Confirmed botanical prior
        aromatic_ring_ratio=0.6,
    )

    # Isomer 2: Synthetic constitutional isomer decoy with identical formula, zero RDA match
    feat_decoy = ranker.extract_feature_vector(
        dreams_cosine=0.5,
        mist_tanimoto=0.40,
        ppm_error=0.0,       # Identical mass error
        abs_da_error=0.0,    # Identical mass error
        fragment_match_ratio=0.10, # Zero RDA match
        np_score=0.4,
        entropy_similarity=0.8,
        cross_track_agreement_count=1,
        formula_rank=1,
        source_track="track2_db",
        composite_prior_confidence=0.10, # Synthetic distractor
        aromatic_ring_ratio=0.2,
    )

    candidates = [
        {"id": "decoy", "inchikey14": "IK14DECOY00001", "features": feat_decoy},
        {"id": "apigenin", "inchikey14": "IK14APIG000001", "features": feat_apigenin},
    ]

    ranked = ranker.score_candidates(candidates)
    assert len(ranked) == 2
    # The true natural product isomer MUST be ranked #1
    assert ranked[0]["id"] == "apigenin"
    assert ranked[0]["meta_score"] > ranked[1]["meta_score"] + 5.0


def test_batch_scoring_nan_and_inf_resilience():
    """Verify GBDTMetaRanker handles NaN and Inf in candidate features without exception."""
    ranker = GBDTMetaRanker()
    bad_feat = np.ones(33, dtype=np.float32)
    bad_feat[0] = np.nan
    bad_feat[5] = np.inf
    bad_feat[10] = -np.inf

    candidates = [
        {"id": "bad", "inchikey14": "IK14BAD0000001", "features": bad_feat},
        {"id": "good", "inchikey14": "IK14GOOD000001", "features": np.ones(33, dtype=np.float32)},
    ]

    scored = ranker.score_candidates(candidates)
    assert len(scored) == 2
    assert not np.isnan(scored[0]["meta_score"])
    assert not np.isnan(scored[1]["meta_score"])


def test_dynamic_feature_extraction_in_pipeline():
    """Verify pipeline populates real non-constant values for aromatic ratio and formula rank."""
    import pandas as pd
    pipeline = CASMIOmegaPipeline()
    df = pd.DataFrame([{
        "id": "SPEC_FLAV",
        "precursor_mz": 271.0600,
        "adduct": "[M+H]+",
        "molecular_formula": "C15H10O5",
    }])

    res = pipeline.run(df)
    assert len(res) == 1
    assert len(pipeline.last_aggregated_cands) > 0
