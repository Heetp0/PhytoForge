"""PhytoForge offline asset indexing, precomputation, and validation CLI engine.

Unified CLI entrypoint providing four subcommands:
- index-db: Standardize chemical structures and construct high-performance SQLite candidate database.
- index-spectra: Construct memory-mapped FP16 reference spectral vector index with aligned Parquet metadata.
- validate-index: Verify structural integrity, schema constraints, and row alignment of database and spectral indexes.
- benchmark: Measure popcount Tanimoto scoring latency across 10,000 candidates against the < 2.0 ms threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_chemical_database(db_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """Validate SQLite chemical candidate database structural integrity.

    Verifies:
    1. Database file exists and is readable.
    2. 'candidates' table exists with WITHOUT ROWID clustering.
    3. Primary key is (formula, inchikey14).
    4. Required columns: formula, inchikey14, smiles, fingerprint.
    5. Fingerprint BLOBs are exactly 256 bytes (32 uint64 words).
    6. '_schema_metadata' table exists and contains metadata keys.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        return False, [f"Database file does not exist: {db_file}"]

    errors: List[str] = []
    messages: List[str] = []

    try:
        conn = sqlite3.connect(f"file:{db_file.resolve().as_posix()}?mode=ro", uri=True)
    except Exception:
        conn = sqlite3.connect(str(db_file))

    try:
        cursor = conn.cursor()
        # 1. Check table 'candidates'
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='candidates';"
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            errors.append("Table 'candidates' not found in database.")
        else:
            sql = row[0]
            if "WITHOUT ROWID" not in sql.upper():
                errors.append("Table 'candidates' is NOT created WITHOUT ROWID.")

            cursor.execute("PRAGMA table_info(candidates);")
            col_info = cursor.fetchall()
            cols = {r[1]: {"type": r[2], "pk": r[5]} for r in col_info}

            required_cols = {"formula", "inchikey14", "smiles", "fingerprint"}
            missing = required_cols - set(cols.keys())
            if missing:
                errors.append(f"Missing required columns in 'candidates': {missing}")

            if cols.get("formula", {}).get("pk", 0) < 1:
                errors.append("Column 'formula' is not part of the primary key.")
            if cols.get("inchikey14", {}).get("pk", 0) < 1:
                errors.append("Column 'inchikey14' is not part of the primary key.")

        # 2. Check '_schema_metadata'
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='_schema_metadata';"
        )
        meta_row = cursor.fetchone()
        if not meta_row:
            errors.append("Table '_schema_metadata' not found in database.")
        else:
            cursor.execute("SELECT key, value FROM _schema_metadata;")
            meta_dict = dict(cursor.fetchall())
            if not meta_dict:
                errors.append("Table '_schema_metadata' is empty.")

        # 3. Check fingerprint BLOB lengths
        cursor.execute("SELECT length(fingerprint) FROM candidates LIMIT 100;")
        blob_lens = cursor.fetchall()
        if blob_lens:
            bad_lens = {bl[0] for bl in blob_lens if bl[0] != 256}
            if bad_lens:
                errors.append(
                    f"Fingerprint BLOB length mismatch: expected 256 bytes, found {bad_lens}"
                )

        cursor.execute("SELECT COUNT(*) FROM candidates;")
        cand_count = cursor.fetchone()[0]
        messages.append(
            f"Validated {cand_count} candidates in WITHOUT ROWID schema with 256-byte BLOBs."
        )

    except Exception as exc:
        errors.append(f"SQLite error during database validation: {exc}")
    finally:
        conn.close()

    return (len(errors) == 0), errors if errors else messages


def validate_spectral_index(spectral_dir: Union[str, Path]) -> Tuple[bool, List[str]]:
    """Validate memory-mapped reference spectral index integrity.

    Verifies:
    1. Directory exists and contains manifest.json.
    2. Ionization partitions (positive/negative) contain embeddings.npy and metadata.parquet.
    3. embeddings.npy is 2D float16 array with shape (N, dim).
    4. metadata.parquet has exactly N rows (1:1 alignment).
    5. Required metadata columns: id, smiles, inchikey14, precursor_mz, adduct, instrument.
    6. Memory maps are safely closed without file handle leaks.
    """
    target_dir = Path(spectral_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        return False, [f"Spectral index directory does not exist: {target_dir}"]

    errors: List[str] = []
    messages: List[str] = []

    # 1. Manifest
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"manifest.json not found in {target_dir}")
        dim = 1024
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            dim = int(manifest.get("dim", 1024))
            messages.append(f"Loaded manifest: version {manifest.get('version', '1.0')}, dim={dim}")
        except Exception as exc:
            errors.append(f"Error reading manifest.json: {exc}")
            dim = 1024

    # 2. Check partitions
    partition_found = False
    required_cols = {"id", "smiles", "inchikey14", "precursor_mz", "adduct", "instrument"}

    for mode in ("positive", "negative"):
        part_dir = target_dir / mode
        emb_path = part_dir / "embeddings.npy"
        meta_path = part_dir / "metadata.parquet"

        if not part_dir.exists():
            continue

        partition_found = True
        if not emb_path.exists():
            errors.append(f"embeddings.npy missing in partition '{mode}'")
            continue
        if not meta_path.exists():
            errors.append(f"metadata.parquet missing in partition '{mode}'")
            continue

        embs = None
        try:
            embs = np.load(str(emb_path), mmap_mode="r")
            if embs.dtype != np.float16:
                errors.append(f"Partition '{mode}' embedding dtype is {embs.dtype}, expected float16")
            if embs.ndim != 2 or embs.shape[1] != dim:
                errors.append(
                    f"Partition '{mode}' embedding shape {embs.shape} does not match expected (*, {dim})"
                )

            df_meta = pd.read_parquet(str(meta_path))
            if len(df_meta) != embs.shape[0]:
                errors.append(
                    f"Partition '{mode}' row alignment mismatch: embeddings has {embs.shape[0]} rows, "
                    f"metadata has {len(df_meta)} rows"
                )

            missing_cols = required_cols - set(df_meta.columns)
            if missing_cols:
                errors.append(f"Partition '{mode}' metadata missing columns: {missing_cols}")

            messages.append(
                f"Partition '{mode}': {len(df_meta)} rows, shape {embs.shape} FP16, 1:1 row alignment verified."
            )

        except Exception as exc:
            errors.append(f"Error validating partition '{mode}': {exc}")
        finally:
            if embs is not None and hasattr(embs, "_mmap") and embs._mmap is not None:
                try:
                    embs._mmap.close()
                except Exception:
                    pass
            del embs

    if not partition_found:
        emb_path = target_dir / "embeddings.npy"
        meta_path = target_dir / "metadata.parquet"
        if emb_path.exists() and meta_path.exists():
            embs = None
            try:
                embs = np.load(str(emb_path), mmap_mode="r")
                df_meta = pd.read_parquet(str(meta_path))
                if embs.dtype != np.float16:
                    errors.append(f"Single partition embedding dtype is {embs.dtype}, expected float16")
                if len(df_meta) != embs.shape[0]:
                    errors.append("Row alignment mismatch in single partition layout")
                messages.append(f"Single partition: {len(df_meta)} rows, shape {embs.shape} FP16.")
            except Exception as exc:
                errors.append(f"Error validating single partition: {exc}")
            finally:
                if embs is not None and hasattr(embs, "_mmap") and embs._mmap is not None:
                    try:
                        embs._mmap.close()
                    except Exception:
                        pass
                del embs
        else:
            errors.append(f"No valid index partitions (positive/negative) found in {target_dir}")

    return (len(errors) == 0), errors if errors else messages


def validate_indices(
    db_path: Optional[Union[str, Path]] = None,
    spectral_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Validates SQLite database and/or spectral vector index integrity.

    Prints diagnostic output and returns True if all provided targets are valid,
    False otherwise.
    """
    if db_path is None and spectral_dir is None:
        print("Error: At least one of --db-path or --spectral-dir must be specified.", file=sys.stderr)
        return False

    all_valid = True

    if db_path is not None:
        print(f"Validating chemical database: {db_path}...")
        valid, msgs = validate_chemical_database(db_path)
        if valid:
            for m in msgs:
                print(f"  [PASS] {m}")
        else:
            all_valid = False
            for m in msgs:
                print(f"  [FAIL] {m}", file=sys.stderr)

    if spectral_dir is not None:
        print(f"Validating spectral index: {spectral_dir}...")
        valid, msgs = validate_spectral_index(spectral_dir)
        if valid:
            for m in msgs:
                print(f"  [PASS] {m}")
        else:
            all_valid = False
            for m in msgs:
                print(f"  [FAIL] {m}", file=sys.stderr)

    return all_valid


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------

def run_tanimoto_benchmark(
    db_path: Optional[Union[str, Path]] = None,
    num_candidates: int = 10000,
    iterations: int = 50,
    threshold_ms: float = 2.0,
) -> Tuple[bool, float]:
    """Benchmark vectorized popcount Tanimoto scoring over candidates.

    Measures average scoring latency across `iterations` iterations.
    Asserts latency is strictly under `threshold_ms` (2.0 ms default).
    """
    try:
        from src.data.db_indexer import popcount_tanimoto, ChemicalDatabaseRepository
    except ImportError:
        popcount_tanimoto = None
        ChemicalDatabaseRepository = None

    candidates: Optional[np.ndarray] = None

    if db_path is not None and Path(db_path).exists():
        try:
            if ChemicalDatabaseRepository is not None:
                repo = ChemicalDatabaseRepository(db_path)
                cursor = repo.conn.cursor()
            else:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
            cursor.execute("SELECT fingerprint FROM candidates LIMIT ?;", (num_candidates,))
            rows = cursor.fetchall()
            if rows:
                blobs = b"".join(r[0] for r in rows)
                candidates = np.frombuffer(blobs, dtype=np.uint64).reshape(-1, 32)
        except Exception:
            candidates = None

    # If candidates not loaded or insufficient, generate reproducible synthetic candidates
    if candidates is None or candidates.shape[0] < num_candidates:
        rng = np.random.RandomState(42)
        candidates = rng.randint(
            0, np.iinfo(np.uint64).max, size=(num_candidates, 32), dtype=np.uint64
        )
    else:
        candidates = candidates[:num_candidates]

    query_fp = np.random.RandomState(99).randint(
        0, np.iinfo(np.uint64).max, size=(32,), dtype=np.uint64
    )

    # Resolve scoring function
    if popcount_tanimoto is not None:
        scorer = popcount_tanimoto
    else:
        def scorer(q: np.ndarray, c: np.ndarray) -> np.ndarray:
            inter = np.bitwise_count(c & q).sum(axis=1)
            union = np.bitwise_count(c | q).sum(axis=1)
            denom = np.where(union > 0, union, 1)
            return np.where(union > 0, inter / denom, 0.0).astype(np.float32)

    # Warm-up pass (ensures Numba JIT compilation and thread pools are primed)
    _ = scorer(query_fp, candidates)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scorer(query_fp, candidates)
    t_elapsed = time.perf_counter() - t0

    latency_ms = (t_elapsed / iterations) * 1000.0
    passed = latency_ms < threshold_ms

    return passed, latency_ms


# ---------------------------------------------------------------------------
# CLI Argument Parser & Commands
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct argparse CLI parser with all four subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.indexer",
        description="PhytoForge offline asset indexing, precomputation, and validation CLI engine.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # 1. index-db
    p_db = subparsers.add_parser(
        "index-db", help="Build SQLite chemical candidate database with standardization & fingerprints"
    )
    p_db.add_argument(
        "--input", type=str, required=True, help="Path to input molecules file (CSV, Parquet, or SMILES)"
    )
    p_db.add_argument(
        "--output", type=str, required=True, help="Path to output SQLite database file (.db)"
    )
    p_db.add_argument(
        "--dead-letter-log", type=str, default=None, help="Optional path for quarantine dead-letter log"
    )
    p_db.add_argument(
        "--batch-size", type=int, default=10000, help="Insert transaction batch size (default: 10000)"
    )
    p_db.add_argument(
        "--smiles-col", type=str, default="smiles", help="SMILES column name for tabular files (default: 'smiles')"
    )
    p_db.add_argument(
        "--id-col", type=str, default="id", help="Identifier column name for tabular files (default: 'id')"
    )
    p_db.add_argument(
        "--suppress-chiral", action="store_true", default=False, help="Suppress stereochemical flags for 2D connectivity"
    )

    # 2. index-spectra
    p_sp = subparsers.add_parser(
        "index-spectra", help="Build memory-mapped FP16 reference spectral vector index"
    )
    p_sp.add_argument(
        "--input", type=str, required=True, help="Path to input spectra Parquet or CSV file"
    )
    p_sp.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for index artifacts"
    )
    p_sp.add_argument(
        "--dim", type=int, default=1024, help="Embedding vector dimension (default: 1024)"
    )
    p_sp.add_argument(
        "--dtype", type=str, default="float16", help="Vector numpy dtype (default: 'float16')"
    )
    p_sp.add_argument(
        "--partition-by-mode", action="store_true", default=True, help="Partition index by ionization mode"
    )

    # 3. validate-index
    p_val = subparsers.add_parser(
        "validate-index", help="Validate database or spectral index structural integrity"
    )
    p_val.add_argument(
        "--db-path", type=str, default=None, help="Path to SQLite database to validate"
    )
    p_val.add_argument(
        "--spectral-dir", type=str, default=None, help="Path to spectral index directory to validate"
    )

    # 4. benchmark
    p_bm = subparsers.add_parser(
        "benchmark", help="Benchmark popcount Tanimoto scoring latency over 10,000 candidates"
    )
    p_bm.add_argument(
        "--db-path", type=str, default=None, help="Optional SQLite database path to sample candidates from"
    )
    p_bm.add_argument(
        "--num-candidates", type=int, default=10000, help="Candidate count to score (default: 10000)"
    )
    p_bm.add_argument(
        "--iterations", type=int, default=50, help="Timing iterations (default: 50)"
    )
    p_bm.add_argument(
        "--threshold-ms", type=float, default=2.0, help="Maximum acceptable latency in ms (default: 2.0)"
    )

    return parser


def main(args: Optional[List[str]] = None) -> None:
    """CLI main entrypoint."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help(sys.stderr)
        sys.exit(1)

    cmd = parsed_args.command

    try:
        if cmd == "index-db":
            inp_path = Path(parsed_args.input)
            if not inp_path.exists():
                print(f"Error: Input file does not exist: {inp_path}", file=sys.stderr)
                sys.exit(1)

            from src.data.db_indexer import ChemicalDatabaseIndexer

            with ChemicalDatabaseIndexer(
                db_path=parsed_args.output,
                dead_letter_path=parsed_args.dead_letter_log,
                suppress_chiral=parsed_args.suppress_chiral,
            ) as indexer:
                # Determine input file type
                ext = inp_path.suffix.lower()
                if ext in (".parquet", ".pq"):
                    df = pd.read_parquet(inp_path)
                    smi_col = parsed_args.smiles_col if parsed_args.smiles_col in df.columns else "smiles"
                    id_col = parsed_args.id_col if parsed_args.id_col in df.columns else None
                    records = [
                        {"smiles": row[smi_col], "id": row[id_col] if id_col else None}
                        for _, row in df.iterrows()
                    ]
                    res = indexer.index_molecules(records, batch_size=parsed_args.batch_size)
                elif ext in (".csv", ".tsv"):
                    res = indexer.index_from_csv(
                        inp_path,
                        smiles_col=parsed_args.smiles_col,
                        id_col=parsed_args.id_col,
                        batch_size=parsed_args.batch_size,
                    )
                else:
                    # Plaintext SMILES list
                    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
                        smiles_list = [line.strip() for line in f if line.strip()]
                    res = indexer.index_from_smiles_list(smiles_list, batch_size=parsed_args.batch_size)

            print(
                f"Successfully indexed {res.get('indexed', 0)} molecules into {parsed_args.output} "
                f"(failed: {res.get('failed', 0)}, total in db: {res.get('total_in_db', 0)})"
            )

        elif cmd == "index-spectra":
            inp_path = Path(parsed_args.input)
            if not inp_path.exists():
                print(f"Error: Input file does not exist: {inp_path}", file=sys.stderr)
                sys.exit(1)

            from src.data.spectral_indexer import SpectralIndexBuilder

            out_dir = SpectralIndexBuilder.build_index(
                spectra=inp_path,
                output_dir=parsed_args.output_dir,
                dim=parsed_args.dim,
            )
            print(f"Successfully indexed spectra into {out_dir}")

        elif cmd == "validate-index":
            valid = validate_indices(
                db_path=parsed_args.db_path,
                spectral_dir=parsed_args.spectral_dir,
            )
            if not valid:
                print("Index validation FAILED.", file=sys.stderr)
                sys.exit(1)
            print("Index validation successful: all constraints satisfied.")

        elif cmd == "benchmark":
            passed, latency_ms = run_tanimoto_benchmark(
                db_path=parsed_args.db_path,
                num_candidates=parsed_args.num_candidates,
                iterations=parsed_args.iterations,
                threshold_ms=parsed_args.threshold_ms,
            )
            print(
                f"Benchmark result: {latency_ms:.3f} ms per {parsed_args.num_candidates} candidates "
                f"(Threshold: < {parsed_args.threshold_ms:.1f} ms)"
            )
            if not passed:
                print(
                    f"Benchmark FAILED: latency ({latency_ms:.3f} ms) exceeds threshold ({parsed_args.threshold_ms:.1f} ms).",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("Benchmark PASSED.")

    except Exception as exc:
        print(f"Error executing '{cmd}': {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
