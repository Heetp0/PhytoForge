"""Synthetic Ranker Dataset Generator for LightGBM LambdaMART training.

Generates 33-dimensional multimodal feature matrices, integer relevance labels,
and query group boundaries for listwise learning-to-rank models.
"""

from typing import List, Tuple
import numpy as np


class RankerDatasetGenerator:
    """Generates synthetic multimodal feature matrices and listwise ranking labels."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_synthetic_ranking_dataset(
        self,
        n_queries: int = 50,
        n_candidates_per_query: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Generate 33-D feature matrix X, relevance labels y in {0, 1, 2, 3}, and query groups.

        Parameters
        ----------
        n_queries : int
            Number of query spectra.
        n_candidates_per_query : int
            Number of candidate structures retrieved per query.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[int]]
            - X: float32 matrix of shape (n_queries * n_candidates_per_query, 33).
            - y: int32 relevance vector of shape (n_queries * n_candidates_per_query,).
            - groups: query group sizes of length n_queries.
        """
        if n_queries <= 0 or n_candidates_per_query <= 0:
            return (
                np.empty((0, 33), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        total_rows = n_queries * n_candidates_per_query
        X = np.zeros((total_rows, 33), dtype=np.float32)
        y = np.zeros(total_rows, dtype=np.int32)
        groups: List[int] = [n_candidates_per_query] * n_queries

        for q_idx in range(n_queries):
            start_row = q_idx * n_candidates_per_query
            true_idx = start_row
            analog_idx = start_row + 1 if n_candidates_per_query > 1 else -1
            weak_indices = (
                [start_row + c for c in range(2, min(4, n_candidates_per_query))]
                if n_candidates_per_query > 2
                else []
            )

            for c_idx in range(n_candidates_per_query):
                curr_row = start_row + c_idx
                # Base random features in [0.0, 0.35]
                feat = self.rng.uniform(0.0, 0.35, size=33).astype(np.float32)

                if curr_row == true_idx:
                    # Ground truth candidate (label = 3)
                    feat[0] = self.rng.uniform(0.85, 0.99)  # dreams_cosine
                    feat[1] = self.rng.uniform(0.80, 0.98)  # entropy_similarity
                    feat[2] = feat[0] * feat[1]            # cross-product
                    feat[3] = self.rng.uniform(0.80, 0.95)  # mist_tanimoto
                    feat[4] = self.rng.uniform(1.20, 2.00)  # np_score
                    feat[5] = feat[3] * feat[4]
                    feat[6] = 1.0                           # mass_ppm_score (0 ppm error)
                    feat[7] = 1.0                           # abs_da_score
                    feat[8] = self.rng.uniform(0.70, 0.95)  # fragment_match_ratio
                    feat[9] = float(self.rng.randint(2, 5)) # cross_track_agreement_count
                    feat[10] = 1.0 if feat[9] >= 2.0 else 0.0
                    feat[11] = 1.0                          # formula_rank = 1
                    y[curr_row] = 3
                elif curr_row == analog_idx:
                    # Near structural analog (label = 2)
                    feat[0] = self.rng.uniform(0.70, 0.85)  # dreams_cosine
                    feat[1] = self.rng.uniform(0.65, 0.80)  # entropy_similarity
                    feat[2] = feat[0] * feat[1]
                    feat[3] = self.rng.uniform(0.60, 0.75)  # mist_tanimoto
                    feat[4] = self.rng.uniform(0.80, 1.30)  # np_score
                    feat[5] = feat[3] * feat[4]
                    feat[6] = 0.8                           # mass_ppm_score
                    feat[7] = 0.8
                    feat[8] = self.rng.uniform(0.50, 0.70)
                    feat[9] = float(self.rng.randint(1, 3))
                    feat[10] = 1.0 if feat[9] >= 2.0 else 0.0
                    feat[11] = 0.5                          # formula_rank = 2
                    y[curr_row] = 2
                elif curr_row in weak_indices:
                    # Weak candidate match (label = 1)
                    feat[0] = self.rng.uniform(0.50, 0.65)
                    feat[1] = self.rng.uniform(0.45, 0.60)
                    feat[2] = feat[0] * feat[1]
                    feat[3] = self.rng.uniform(0.40, 0.55)
                    feat[4] = self.rng.uniform(0.40, 0.80)
                    feat[5] = feat[3] * feat[4]
                    feat[6] = 0.5
                    feat[7] = 0.5
                    feat[8] = self.rng.uniform(0.25, 0.45)
                    feat[9] = 1.0
                    feat[10] = 0.0
                    feat[11] = 0.33
                    y[curr_row] = 1
                else:
                    # Noise / distractor candidate (label = 0)
                    feat[2] = feat[0] * feat[1]
                    feat[5] = feat[3] * feat[4]
                    feat[10] = 0.0
                    feat[11] = 0.1
                    y[curr_row] = 0

                # Clear track one-hot slots (12, 13, 14, 15, 32)
                for tidx in [12, 13, 14, 15, 32]:
                    feat[tidx] = 0.0
                # Assign one active track
                track_idx = int(self.rng.choice([12, 13, 14, 15, 32]))
                feat[track_idx] = 1.0

                X[curr_row] = feat

        assert not np.isnan(X).any(), "NaN found in generated feature matrix X"
        assert not np.isinf(X).any(), "Inf found in generated feature matrix X"
        return X, y, groups
