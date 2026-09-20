from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class CandidateFeatureVector:
    features: np.ndarray


class GBDTMetaRanker:
    """Module 4: 33-Feature GBDT LambdaMART Meta-Ranker."""

    def __init__(self, weights: Optional[np.ndarray] = None):
        # Default heuristic weights if offline model not loaded
        self.weights = weights if weights is not None else np.ones(33, dtype=np.float32)

    def extract_feature_vector(
        self,
        dreams_cosine: float,
        mist_tanimoto: float,
        ppm_error: float,
        abs_da_error: float,
        fragment_match_ratio: float,
        np_score: float,
        entropy_similarity: float,
        cross_track_agreement_count: int,
        formula_rank: int,
        source_track: str,
        score_delta_to_top2: float = 0.05,
        track1_score_margin: float = 0.1,
        track2_tanimoto_margin: float = 0.1,
        cross_track_consensus_ratio: float = 0.5,
        molecular_weight_norm: float = 0.4,
        h_bond_donors_est: float = 2.0,
        h_bond_acceptors_est: float = 4.0,
        rotatable_bonds_est: float = 3.0,
        tpsa_est_norm: float = 0.35,
        aromatic_ring_ratio: float = 0.5,
        spectral_entropy_delta: float = 0.02,
        explained_intensity_ratio: float = 0.85,
        neutral_loss_match_count: int = 2,
        precursor_residual_intensity: float = 0.05,
        unassigned_peak_ratio: float = 0.15,
        composite_prior_confidence: float = 0.75,
    ) -> np.ndarray:
        vec = np.zeros(33, dtype=np.float32)
        # 1. Spectral similarities
        vec[0] = dreams_cosine
        vec[1] = entropy_similarity
        vec[2] = dreams_cosine * entropy_similarity
        # 2. Structural fit
        vec[3] = mist_tanimoto
        vec[4] = np_score
        vec[5] = mist_tanimoto * np_score
        # 3. Mass accuracy
        vec[6] = 1.0 / (1.0 + abs(ppm_error))
        vec[7] = 1.0 / (1.0 + abs(abs_da_error) * 100.0)
        # 4. Fragmentation
        vec[8] = fragment_match_ratio
        # 5. Cross-track agreement
        vec[9] = float(cross_track_agreement_count)
        vec[10] = 1.0 if cross_track_agreement_count >= 2 else 0.0
        # 6. Formula rank
        vec[11] = 1.0 / float(max(1, formula_rank))
        # 7. One-hot source tracks (indices 12-15, and index 32 for track_knapsack)
        track_map = {
            "track1_dreams": 12,
            "track2_db": 13,
            "track3_denovo": 14,
            "track_network": 15,
            "track_knapsack": 32,
        }
        if source_track in track_map:
            vec[track_map[source_track]] = 1.0
        # 8. Cross-track score deltas & margins (indices 16-19)
        vec[16] = score_delta_to_top2
        vec[17] = track1_score_margin
        vec[18] = track2_tanimoto_margin
        vec[19] = cross_track_consensus_ratio
        # 9. Physicochemical & topological descriptors (indices 20-25)
        vec[20] = molecular_weight_norm
        vec[21] = h_bond_donors_est / 10.0
        vec[22] = h_bond_acceptors_est / 15.0
        vec[23] = rotatable_bonds_est / 12.0
        vec[24] = tpsa_est_norm
        vec[25] = aromatic_ring_ratio
        # 10. Advanced spectral & neutral loss features (indices 26-31)
        vec[26] = spectral_entropy_delta
        vec[27] = explained_intensity_ratio
        vec[28] = min(1.0, neutral_loss_match_count / 5.0)
        vec[29] = precursor_residual_intensity
        vec[30] = unassigned_peak_ratio
        vec[31] = composite_prior_confidence
        return vec

    def score_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for cand in candidates:
            feat = cand["features"]
            score_val = float(
                np.dot(feat[: len(self.weights)], self.weights[: len(feat)])
            )
            cand["meta_score"] = score_val
            cand["score"] = score_val
        candidates.sort(key=lambda x: x["meta_score"], reverse=True)
        return candidates
