"""CLI and module to train LightGBM LambdaMART meta-ranker on authentic natural product spectra."""

import argparse
from pathlib import Path
import pickle
import time
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.ranker_dataset import AuthenticRankerDatasetBuilder
from src.reranking.meta_ranker import GBDTMetaRanker
from src.reranking.trainer import save_booster, train_lambda_mart
from src.retrieval.database_search import SoftDatabaseSearcher


def train_meta_ranker_from_parquet(
    train_parquet_path: Path,
    db_path: Path,
    output_model_path: Path,
    val_parquet_path: Optional[Path] = None,
    fragment_library_path: Optional[Path] = None,
    n_queries: int = 350,
    candidates_per_query: int = 25,
    n_synthetic_queries: int = 150,
    n_estimators: int = 75,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42,
) -> Path:
    """Extract authentic training pairs, train LightGBM LambdaMART, and save booster."""
    t0 = time.time()
    holdout_iks: Set[str] = set()
    if val_parquet_path is not None and val_parquet_path.exists():
        val_df = pd.read_parquet(val_parquet_path)
        if "inchikey14" in val_df.columns:
            holdout_iks = set(str(x).strip() for x in val_df["inchikey14"] if pd.notna(x))
        print(f"Loaded {len(holdout_iks)} holdout InChIKey14s from {val_parquet_path} for leakage prevention.")

    frag_lib: Optional[Dict[int, List[str]]] = None
    if fragment_library_path is not None and fragment_library_path.exists():
        with open(fragment_library_path, "rb") as f:
            frag_lib = pickle.load(f)
        print(f"Loaded fragment library with {len(frag_lib)} neutral loss entries.")

    print(f"Connecting to chemical candidate database at {db_path}...")
    db_searcher = SoftDatabaseSearcher(db_path=db_path)

    builder = AuthenticRankerDatasetBuilder(
        db_path=db_path,
        meta_ranker=GBDTMetaRanker(),
        random_state=random_state,
    )

    print(f"Reading training rows from {train_parquet_path}...")
    pq_file = pq.ParquetFile(train_parquet_path)
    # Read first 1 or 2 row groups to get enough diverse natural product queries
    cols = [
        "normalized_smiles",
        "inchikey14",
        "molecular_formula",
        "precursor_mz",
        "adduct",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]
    train_df = pq_file.read_row_group(0, columns=cols).to_pandas()
    print(f"Loaded {len(train_df)} candidate training rows from row group 0.")

    print(f"Extracting authentic candidate features for up to {n_queries} queries (pool size {candidates_per_query})...")
    X, y, groups = builder.build_dataset_from_dataframe(
        df=train_df,
        db_searcher=db_searcher,
        n_queries=n_queries,
        candidates_per_query=candidates_per_query,
        holdout_inchikey14s=holdout_iks,
        fragment_library=frag_lib,
    )
    db_searcher.close()

    # Blend with synthetic physics-grounded queries if requested
    if n_synthetic_queries > 0:
        from src.data.ranker_dataset import RankerDatasetGenerator
        synth_gen = RankerDatasetGenerator(random_state=random_state)
        X_s, y_s, groups_s = synth_gen.generate_synthetic_ranking_dataset(
            n_queries=n_synthetic_queries,
            n_candidates_per_query=candidates_per_query,
        )
        if len(groups_s) > 0:
            X = np.vstack([X, X_s])
            y = np.concatenate([y, y_s])
            groups = groups + groups_s
            print(f"Blended with {n_synthetic_queries} synthetic queries ({len(y_s)} rows).")

    print(f"Dataset extracted: {X.shape[0]} total candidate rows across {len(groups)} queries.")
    print(f"Label distribution: 3 (Ground truth): {(y == 3).sum()}, 2 (High analog): {(y == 2).sum()}, 1 (Low analog): {(y == 1).sum()}, 0 (Distractor): {(y == 0).sum()}")

    # Split queries 80% train / 20% validation
    n_total_queries = len(groups)
    n_train_queries = max(1, int(n_total_queries * 0.8))
    train_groups = groups[:n_train_queries]
    val_groups = groups[n_train_queries:]

    n_train_rows = sum(train_groups)
    X_train, y_train = X[:n_train_rows], y[:n_train_rows]
    X_val, y_val = X[n_train_rows:], y[n_train_rows:]

    eval_data = (X_val, y_val, val_groups) if len(val_groups) > 0 else None

    print(f"Training LightGBM LambdaMART: {n_train_queries} train queries, {len(val_groups)} val queries, {n_estimators} trees...")
    booster = train_lambda_mart(
        X=X_train,
        y=y_train,
        groups=train_groups,
        eval_data=eval_data,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
    )

    out_p = save_booster(booster, output_model_path)
    elapsed = time.time() - t0
    print(f"Meta-ranker successfully trained and saved to {out_p} in {elapsed:.1f}s.")
    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaMART ranker on authentic natural product spectra.")
    parser.add_argument("--train-parquet", type=Path, default=Path("data/train.parquet"), help="Path to train.parquet")
    parser.add_argument("--val-parquet", type=Path, default=Path("data/val_benchmark_sample.parquet"), help="Path to validation sample parquet")
    parser.add_argument("--db-path", type=Path, default=Path("data/candidates.db"), help="Path to candidate SQLite DB")
    parser.add_argument("--fragment-library-path", type=Path, default=Path("models/fragment_lib.pkl"), help="Path to fragment lib")
    parser.add_argument("--output-model", type=Path, default=Path("models/meta_ranker.txt"), help="Path to save trained booster")
    parser.add_argument("--n-queries", type=int, default=300, help="Number of authentic queries to train on")
    parser.add_argument("--n-synthetic-queries", type=int, default=150, help="Number of synthetic domain queries to blend")
    parser.add_argument("--candidates-per-query", type=int, default=25, help="Candidate pool size per query")
    parser.add_argument("--n-estimators", type=int, default=75, help="Number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=31, help="Max leaves per tree")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_meta_ranker_from_parquet(
        train_parquet_path=args.train_parquet,
        db_path=args.db_path,
        output_model_path=args.output_model,
        val_parquet_path=args.val_parquet,
        fragment_library_path=args.fragment_library_path,
        n_queries=args.n_queries,
        candidates_per_query=args.candidates_per_query,
        n_synthetic_queries=args.n_synthetic_queries,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
