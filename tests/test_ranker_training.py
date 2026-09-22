"""Unit and integration tests for LightGBM LambdaMART training, dataset generation, and GBDTMetaRanker vectorized batch inference."""

import time
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from src.data.ranker_dataset import RankerDatasetGenerator
from src.reranking.meta_ranker import GBDTMetaRanker
from src.reranking.trainer import load_booster, save_booster, train_lambda_mart


class TestRankerDatasetGenerator:
    """Tests for RankerDatasetGenerator."""

    def test_generate_synthetic_ranking_dataset_shapes_and_types(self):
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(
            n_queries=15, n_candidates_per_query=20
        )
        assert X.shape == (300, 33)
        assert X.dtype == np.float32
        assert y.shape == (300,)
        assert y.dtype == np.int32
        assert len(groups) == 15
        assert sum(groups) == 300
        assert all(g == 20 for g in groups)

        # Labels must be valid relevance grades in {0, 1, 2, 3}
        unique_labels = set(np.unique(y))
        assert unique_labels.issubset({0, 1, 2, 3})
        # All 4 levels should be present with 20 candidates per query
        assert unique_labels == {0, 1, 2, 3}

        # Check absence of NaNs and Infs
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_reproducibility(self):
        gen1 = RankerDatasetGenerator(random_state=123)
        X1, y1, groups1 = gen1.generate_synthetic_ranking_dataset(n_queries=5, n_candidates_per_query=10)

        gen2 = RankerDatasetGenerator(random_state=123)
        X2, y2, groups2 = gen2.generate_synthetic_ranking_dataset(n_queries=5, n_candidates_per_query=10)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert groups1 == groups2

        gen3 = RankerDatasetGenerator(random_state=456)
        X3, y3, _ = gen3.generate_synthetic_ranking_dataset(n_queries=5, n_candidates_per_query=10)
        assert not np.array_equal(X1, X3)

    def test_edge_cases_empty_and_small(self):
        gen = RankerDatasetGenerator(random_state=42)
        # Zero queries
        X_empty, y_empty, groups_empty = gen.generate_synthetic_ranking_dataset(0, 20)
        assert X_empty.shape == (0, 33)
        assert len(y_empty) == 0
        assert groups_empty == []

        # Zero candidates
        X_zero_cands, y_zero_cands, groups_zero_cands = gen.generate_synthetic_ranking_dataset(5, 0)
        assert X_zero_cands.shape == (0, 33)
        assert len(y_zero_cands) == 0
        assert groups_zero_cands == []

        # 1 query, 1 candidate
        X_1, y_1, groups_1 = gen.generate_synthetic_ranking_dataset(1, 1)
        assert X_1.shape == (1, 33)
        assert y_1[0] == 3
        assert groups_1 == [1]

    def test_feature_hierarchy(self):
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=5, n_candidates_per_query=20)
        # Query 0: ground truth at index 0, analog at index 1
        gt_dreams = X[0, 0]
        analog_dreams = X[1, 0]
        assert gt_dreams >= 0.85
        assert analog_dreams >= 0.70
        assert X[0, 6] == 1.0  # Ground truth perfect mass score
        assert X[1, 6] == 0.8  # Analog high mass score


class TestLambdaMARTTraining:
    """Tests for train_lambda_mart, save_booster, and load_booster."""

    def test_train_lambda_mart_basic(self):
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=20, n_candidates_per_query=10)

        booster = train_lambda_mart(
            X,
            y,
            groups,
            n_estimators=10,
            learning_rate=0.1,
            num_leaves=15,
            random_state=42,
        )
        assert booster is not None
        preds = booster.predict(X[:10])
        assert len(preds) == 10
        # The ground truth candidate (idx 0) should score higher than distractors
        assert preds[0] > preds[-1]

    def test_train_lambda_mart_with_eval_data(self):
        gen = RankerDatasetGenerator(random_state=42)
        X_train, y_train, groups_train = gen.generate_synthetic_ranking_dataset(n_queries=20, n_candidates_per_query=10)
        X_val, y_val, groups_val = gen.generate_synthetic_ranking_dataset(n_queries=5, n_candidates_per_query=10)

        booster = train_lambda_mart(
            X_train,
            y_train,
            groups_train,
            eval_data=(X_val, y_val, groups_val),
            n_estimators=15,
            learning_rate=0.08,
            random_state=42,
        )
        assert booster is not None
        preds_val = booster.predict(X_val)
        assert len(preds_val) == len(X_val)

    def test_native_txt_save_load_roundtrip(self, tmp_path: Path):
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=15, n_candidates_per_query=10)
        booster = train_lambda_mart(X, y, groups, n_estimators=10, random_state=42)

        model_path = tmp_path / "models" / "lambda_mart_ranker.txt"
        saved_path = save_booster(booster, model_path)
        assert saved_path == model_path
        assert model_path.exists()
        assert model_path.stat().st_size > 0

        # Verify it is native text format and NOT a binary pickle file
        with open(model_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            assert "tree" in first_line.lower() or "version" in first_line.lower() or "=" in first_line

        # Load back and verify exact prediction equivalence
        loaded_booster = load_booster(model_path)
        preds_orig = booster.predict(X[:25])
        preds_loaded = loaded_booster.predict(X[:25])
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5, atol=1e-6)

    def test_load_booster_nonexistent_file_raises_error(self, tmp_path: Path):
        missing_path = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_booster(missing_path)


class TestGBDTMetaRankerInference:
    """Tests for GBDTMetaRanker with booster integration and backward compatibility."""

    def test_meta_ranker_with_trained_booster(self):
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=15, n_candidates_per_query=10)
        booster = train_lambda_mart(X, y, groups, n_estimators=10, random_state=42)

        ranker = GBDTMetaRanker(booster=booster)
        assert ranker.booster is booster

        # Construct candidates: C0 is ground truth (X[0]), C1 is analog (X[1]), C9 is noise (X[9])
        candidates = [
            {"id": "cand_noise", "features": X[9]},
            {"id": "cand_gt", "features": X[0]},
            {"id": "cand_analog", "features": X[1]},
        ]
        scored = ranker.score_candidates(candidates)
        assert len(scored) == 3
        # Ground truth candidate must rank top
        assert scored[0]["id"] == "cand_gt"
        assert scored[0]["meta_score"] >= scored[1]["meta_score"] >= scored[2]["meta_score"]
        # Verify score field is also populated
        assert scored[0]["score"] == scored[0]["meta_score"]

    def test_meta_ranker_init_with_model_path(self, tmp_path: Path):
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=10, n_candidates_per_query=10)
        booster = train_lambda_mart(X, y, groups, n_estimators=10, random_state=42)

        model_file = tmp_path / "model.txt"
        save_booster(booster, model_file)

        ranker = GBDTMetaRanker(model_path=model_file)
        assert ranker.booster is not None

        cands = [
            {"id": "A", "features": X[5]},
            {"id": "B", "features": X[0]},
        ]
        scored = ranker.score_candidates(cands)
        assert len(scored) == 2
        assert scored[0]["id"] == "B"

    def test_meta_ranker_model_path_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            GBDTMetaRanker(model_path=tmp_path / "missing_ranker.txt")

    def test_meta_ranker_empty_candidates_list(self):
        ranker = GBDTMetaRanker()
        assert ranker.score_candidates([]) == []

        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=5, n_candidates_per_query=5)
        booster = train_lambda_mart(X, y, groups, n_estimators=5)
        ranker_booster = GBDTMetaRanker(booster=booster)
        assert ranker_booster.score_candidates([]) == []

    def test_backward_compatibility_none_booster(self):
        ranker = GBDTMetaRanker(booster=None)
        cands = [
            {"id": "low", "features": np.zeros(33, dtype=np.float32)},
            {"id": "high", "features": np.ones(33, dtype=np.float32)},
        ]
        scored = ranker.score_candidates(cands)
        assert scored[0]["id"] == "high"
        assert scored[0]["meta_score"] == 33.0
        assert scored[1]["id"] == "low"
        assert scored[1]["meta_score"] == 0.0

    def test_backward_compatibility_custom_weights(self):
        weights = np.zeros(33, dtype=np.float32)
        weights[0] = 5.0
        ranker = GBDTMetaRanker(weights=weights, booster=None)

        cand1 = {"id": "C1", "features": np.zeros(33, dtype=np.float32)}
        cand1["features"][0] = 0.8
        cand2 = {"id": "C2", "features": np.zeros(33, dtype=np.float32)}
        cand2["features"][0] = 0.2

        scored = ranker.score_candidates([cand2, cand1])
        assert scored[0]["id"] == "C1"
        assert pytest.approx(scored[0]["meta_score"], 1e-4) == 4.0
        assert pytest.approx(scored[1]["meta_score"], 1e-4) == 1.0

    def test_mock_booster_fallback_num_threads_type_error(self):
        class MockBooster:
            def predict(self, X: np.ndarray) -> np.ndarray:
                # Does not accept num_threads keyword argument
                return np.sum(X, axis=1)

        ranker = GBDTMetaRanker(booster=MockBooster())
        cands = [
            {"id": "A", "features": np.ones(33, dtype=np.float32)},
            {"id": "B", "features": np.zeros(33, dtype=np.float32)},
        ]
        scored = ranker.score_candidates(cands)
        assert scored[0]["id"] == "A"
        assert scored[0]["meta_score"] == 33.0

    def test_vectorized_batch_scoring_latency_sub_1ms(self):
        """CRITICAL: Batch prediction must achieve < 1.0 ms for 25-100 candidates."""
        gen = RankerDatasetGenerator(random_state=42)
        X, y, groups = gen.generate_synthetic_ranking_dataset(n_queries=25, n_candidates_per_query=20)
        booster = train_lambda_mart(X, y, groups, n_estimators=25, random_state=42)
        ranker = GBDTMetaRanker(booster=booster)

        # Create candidate set of 50 candidates (typical retrieval size)
        candidates = [{"id": f"cand_{i}", "features": X[i]} for i in range(50)]

        # Warm-up runs
        for _ in range(5):
            ranker.score_candidates(candidates)

        # Timed benchmark: 30 iterations
        n_iters = 30
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ranker.score_candidates(candidates)
        elapsed = time.perf_counter() - t0
        avg_latency_ms = (elapsed / n_iters) * 1000.0

        # Assert strictly sub-1.0 ms
        assert avg_latency_ms < 1.0, f"Average latency was {avg_latency_ms:.3f} ms, expected < 1.0 ms"
