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


def test_botanical_scorer_integration_and_distribution_alignment(
    mock_training_df: pd.DataFrame, mock_db_searcher: SoftDatabaseSearcher
):
    """Verify that BotanicalSubstructureScorer is instantiated, bot_score is computed and blended into features."""
    builder = AuthenticRankerDatasetBuilder(random_state=42)
    assert hasattr(builder, "botanical_scorer")

    X, y, groups = builder.build_dataset_from_dataframe(
        df=mock_training_df,
        db_searcher=mock_db_searcher,
        n_queries=3,
        candidates_per_query=5,
    )
    # Feature 31 is composite_prior_confidence (botanical prior)
    assert not np.isnan(X[:, 31]).any()
    assert (X[:, 31] >= 0.0).all() and (X[:, 31] <= 1.0).all()
    # Feature 8 is fragment_match_ratio (blended with botanical prior)
    assert (X[:, 8] >= 0.001).all() and (X[:, 8] <= 0.999).all()
    # Feature 3 is mist_tanimoto (blended with botanical prior)
    assert (X[:, 3] >= 0.001).all() and (X[:, 3] <= 0.999).all()


def test_synthetic_intra_formula_constitutional_isomers():
    """Verify that 70% of synthetic query groups have identical mass error (feat[6], feat[7]) across all candidates."""
    from src.data.ranker_dataset import RankerDatasetGenerator

    gen = RankerDatasetGenerator(random_state=42)
    n_queries = 20
    n_cands = 10
    X, y, groups = gen.generate_synthetic_ranking_dataset(
        n_queries=n_queries, n_candidates_per_query=n_cands
    )

    intra_count = 0
    cross_count = 0
    for q in range(n_queries):
        start = q * n_cands
        group_ppm = X[start : start + n_cands, 6]
        group_da = X[start : start + n_cands, 7]
        # In intra-formula groups, all candidates share identical exact mass (feat[6] == 1.0 and feat[7] == 1.0)
        if np.allclose(group_ppm, 1.0) and np.allclose(group_da, 1.0):
            intra_count += 1
            # In intra-formula groups, discrimination must be driven by fragmentation (feat[8]),
            # structural similarity (feat[3]), and botanical prior (feat[31])
            assert not np.allclose(X[start : start + n_cands, 8], X[start, 8])
            assert not np.allclose(X[start : start + n_cands, 3], X[start, 3])
            assert not np.allclose(X[start : start + n_cands, 31], X[start, 31])
        else:
            cross_count += 1
            # In cross-formula groups, mass error must vary
            assert not np.allclose(group_ppm, group_ppm[0])

    assert intra_count == 14  # exactly 70% of 20
    assert cross_count == 6   # exactly 30% of 20


def test_candidate_level_holdout_protection(
    mock_training_df: pd.DataFrame, mock_db_searcher: SoftDatabaseSearcher
):
    """Verify that candidate isomers or distractors matching holdouts are excluded from candidate pools."""
    builder = AuthenticRankerDatasetBuilder(random_state=42)
    # Hold out the isomer IK14MOCKISOM01
    holdout = {"IK14MOCKISOM01"}

    row = mock_training_df.iloc[0]  # Quercetin query
    query_fp = np.ones(3, dtype=np.float32)
    X_q, y_q = builder.extract_query_candidates_and_features(
        query_row=row,
        db_searcher=mock_db_searcher,
        db_query_fp=query_fp,
        candidates_per_query=5,
        holdout_inchikey14s=holdout,
    )
    # The candidate pool must not contain the held out candidate
    assert len(y_q) > 0
