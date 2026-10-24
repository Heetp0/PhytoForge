"""Tests for AuthenticRankerDatasetBuilder and authentic LambdaMART training."""

import numpy as np
import pandas as pd
import pytest
from src.data.ranker_dataset import AuthenticRankerDatasetBuilder
from src.reranking.meta_ranker import GBDTMetaRanker
from src.reranking.trainer import train_lambda_mart, save_booster, load_booster
from src.retrieval.database_search import SoftDatabaseSearcher


@pytest.fixture
def mock_training_df() -> pd.DataFrame:
    """Fixture providing mock authentic training spectra queries."""
    return pd.DataFrame([
        {
            "normalized_smiles": "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12",  # Quercetin
            "inchikey14": "REFJWTPEDVJJIY",
            "molecular_formula": "C15H10O7",
            "precursor_mz": 303.0500,
            "adduct": "[M+H]+",
            "ms2_mzs": [303.05, 153.02, 137.02, 69.01],
            "ms2_normalized_intensities": [1.0, 0.6, 0.4, 0.2],
        },
        {
            "normalized_smiles": "O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12",  # Apigenin
            "inchikey14": "KZNIFHPLKGYRTM",
            "molecular_formula": "C15H10O5",
            "precursor_mz": 271.0600,
            "adduct": "[M+H]+",
            "ms2_mzs": [271.06, 153.02, 119.05],
            "ms2_normalized_intensities": [1.0, 0.5, 0.3],
        },
        {
            "normalized_smiles": "c1ccccc1O",  # Phenol
            "inchikey14": "ISWSIDIOOBJBQZ",
            "molecular_formula": "C6H6O",
            "precursor_mz": 95.0491,
            "adduct": "[M+H]+",
            "ms2_mzs": [95.0491, 67.05],
            "ms2_normalized_intensities": [1.0, 0.8],
        },
    ])


@pytest.fixture
def mock_db_searcher() -> SoftDatabaseSearcher:
    """In-memory SoftDatabaseSearcher with candidate structures."""
    db = {
        "C15H10O7": [
            ("O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "REFJWTPEDVJJIY", np.ones(3, dtype=np.float32)),
            ("O=C1C(O)=C(c2cc(O)c(O)c(O)c2)Oc2cc(O)ccc12", "IK14MOCKISOM01", np.ones(3, dtype=np.float32)),
        ],
        "C15H10O5": [
            ("O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "KZNIFHPLKGYRTM", np.ones(3, dtype=np.float32)),
            ("O=C1C(=Cc2ccc(O)cc2)Oc2cc(O)cc(O)c12", "ACGXIFQKQYZLOU", np.ones(3, dtype=np.float32)),
        ],
        "C6H6O": [
            ("c1ccccc1O", "ISWSIDIOOBJBQZ", np.ones(3, dtype=np.float32)),
            ("c1ccncc1", "IK14MOCKDIST01", np.ones(3, dtype=np.float32)),
        ],
    }
    return SoftDatabaseSearcher(db=db)


def test_authentic_dataset_builder_basic(mock_training_df: pd.DataFrame, mock_db_searcher: SoftDatabaseSearcher):
    """Test extracting authentic feature matrices and relevance labels."""
    builder = AuthenticRankerDatasetBuilder(random_state=42)
    X, y, groups = builder.build_dataset_from_dataframe(
        df=mock_training_df,
        db_searcher=mock_db_searcher,
        n_queries=3,
        candidates_per_query=5,
    )

    assert X.shape[1] == 33
    assert len(y) == X.shape[0]
    assert len(groups) == 3
    assert sum(groups) == len(y)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

    # Verify relevance labels in {0, 1, 2, 3} and ground truth has 3
    assert set(np.unique(y)).issubset({0, 1, 2, 3})
    assert 3 in y


def test_authentic_dataset_builder_holdouts(mock_training_df: pd.DataFrame, mock_db_searcher: SoftDatabaseSearcher):
    """Verify that queries matching holdout InChIKey14s are strictly excluded."""
    builder = AuthenticRankerDatasetBuilder(random_state=42)
    holdout = {"REFJWTPEDVJJIY"}  # Hold out Quercetin

    X, y, groups = builder.build_dataset_from_dataframe(
        df=mock_training_df,
        db_searcher=mock_db_searcher,
        n_queries=3,
        candidates_per_query=5,
        holdout_inchikey14s=holdout,
    )

    assert len(groups) == 2  # Quercetin excluded, Apigenin & Phenol retained


def test_authentic_train_and_predict(mock_training_df: pd.DataFrame, mock_db_searcher: SoftDatabaseSearcher, tmp_path):
    """End-to-end integration test: build authentic data, train LightGBM, save, load, and score."""
    builder = AuthenticRankerDatasetBuilder(random_state=42)
    X, y, groups = builder.build_dataset_from_dataframe(
        df=mock_training_df,
        db_searcher=mock_db_searcher,
        n_queries=3,
        candidates_per_query=5,
    )

    booster = train_lambda_mart(
        X,
        y,
        groups,
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=7,
        random_state=42,
    )
    assert booster is not None

    model_path = tmp_path / "meta_ranker_test.txt"
    save_booster(booster, model_path)
    assert model_path.exists()

    loaded_ranker = GBDTMetaRanker(model_path=model_path)
    test_cands = [
        {"inchikey14": "REFJWTPEDVJJIY", "features": X[0]},
        {"inchikey14": "IK14MOCKISOM01", "features": X[1]},
    ]
    scored = loaded_ranker.score_candidates(test_cands)
    assert len(scored) == 2
    assert "meta_score" in scored[0]
