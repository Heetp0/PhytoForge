"""LightGBM LambdaMART ranker training and native model persistence.

Provides end-to-end listwise ranking model training using LightGBM's
LambdaMART engine with native text-based serialization (zero pickle).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    _LIGHTGBM_AVAILABLE = False


def train_lambda_mart(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[int],
    eval_data: Optional[Tuple[np.ndarray, np.ndarray, List[int]]] = None,
    params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42,
) -> Any:
    """Train LightGBM LambdaMART ranker using lambdarank objective and NDCG metric.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, num_features).
    y : np.ndarray
        Integer relevance labels of shape (N,).
    groups : List[int]
        Number of candidates per query spectrum.
    eval_data : Optional[Tuple[np.ndarray, np.ndarray, List[int]]]
        Validation tuple (X_val, y_val, groups_val).
    params : Optional[Dict[str, Any]]
        Optional overrides for LightGBM parameters.
    n_estimators : int
        Number of boosting rounds (trees).
    learning_rate : float
        Boosting learning rate.
    num_leaves : int
        Maximum leaves per tree.
    random_state : int
        Random seed for tree construction.

    Returns
    -------
    lgb.Booster
        Trained LightGBM Booster object.
    """
    if not _LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed. Install lightgbm to train meta-ranker.")

    train_ds = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
    valid_sets = [train_ds]
    valid_names = ["train"]

    if eval_data is not None:
        X_val, y_val, groups_val = eval_data
        val_ds = lgb.Dataset(
            X_val,
            label=y_val,
            group=groups_val,
            reference=train_ds,
            free_raw_data=False,
        )
        valid_sets.append(val_ds)
        valid_names.append("valid")

    default_params: Dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1, 5, 10, 25],
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "random_state": random_state,
        "n_jobs": 2,
    }

    num_boost_round = n_estimators
    if params is not None:
        user_params = dict(params)
        if "n_estimators" in user_params:
            num_boost_round = int(user_params.pop("n_estimators"))
        if "num_boost_round" in user_params:
            num_boost_round = int(user_params.pop("num_boost_round"))
        default_params.update(user_params)

    booster = lgb.train(
        default_params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
    )
    return booster


def save_booster(booster: Any, path: Union[str, Path]) -> Path:
    """Save booster model as native LightGBM text representation (zero pickle).

    Parameters
    ----------
    booster : lgb.Booster
        Trained LightGBM booster object.
    path : Union[str, Path]
        Target file path.

    Returns
    -------
    Path
        Absolute or resolved Path to the saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(p))
    return p


def load_booster(path: Union[str, Path]) -> Any:
    """Load native LightGBM text model.

    Parameters
    ----------
    path : Union[str, Path]
        Path to native LightGBM model text file.

    Returns
    -------
    lgb.Booster
        Loaded LightGBM booster object.
    """
    if not _LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is required to load a LightGBM booster model.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return lgb.Booster(model_file=str(p))
