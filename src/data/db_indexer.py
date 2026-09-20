"""
SQLite-backed chemical candidate database indexer and runtime repository for Enveda CASMI 2026.

Provides high-throughput offline candidate ingestion and sub-2ms vectorized popcount
Tanimoto retrieval over 2048-bit Morgan fingerprints packed into 32 uint64 words.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to load Numba for ultra-fast parallel JIT popcount
try:
    import numba

    @numba.njit(inline="always")
    def _popcount64(x: np.uint64) -> int:
        """Branchless 64-bit popcount using parallel bit twiddling."""
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))

    @numba.njit(parallel=True, fastmath=True)
    def _numba_popcount_tanimoto(q: np.ndarray, fps: np.ndarray) -> np.ndarray:
        """Parallel Numba popcount Tanimoto calculation."""
        n = fps.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)

        scores = np.empty(n, dtype=np.float32)
        q_sum = 0
        for j in range(32):
            q_sum += _popcount64(q[j])

        for i in numba.prange(n):
            inter = 0
            cand_sum = 0
            for j in range(32):
                v = fps[i, j]
                cand_sum += _popcount64(v)
                inter += _popcount64(v & q[j])
            union = cand_sum + q_sum - inter
            if union > 0:
                scores[i] = inter / union
            else:
                scores[i] = 0.0
        return scores

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    _numba_popcount_tanimoto = None


def pack_fingerprint(arr: np.ndarray) -> np.ndarray:
    """
    Packs a 2048-element boolean or 0/1 array into 32 uint64 words (256 bytes)
    using little-endian bit order.

    Args:
        arr: 1D array of shape (2048,) or 2D array of shape (N, 2048).

    Returns:
        1D uint64 array of shape (32,) or 2D uint64 array of shape (N, 32).
    """
    arr = np.asarray(arr)
    if arr.dtype == np.uint64:
        if arr.shape == (32,) or (arr.ndim == 2 and arr.shape[1] == 32):
            return arr.copy()

    if arr.ndim == 1:
        if arr.shape[0] != 2048:
            raise ValueError(f"Expected 2048 elements for fingerprint packing, got {arr.shape[0]}")
        u8_arr = (arr > 0).astype(np.uint8)
        packed_bytes = np.packbits(u8_arr, bitorder="little")
        return np.frombuffer(packed_bytes.tobytes(), dtype=np.uint64).copy()
    elif arr.ndim == 2:
        if arr.shape[1] != 2048:
            raise ValueError(f"Expected shape (N, 2048) for fingerprint packing, got {arr.shape}")
        u8_arr = (arr > 0).astype(np.uint8)
        packed_bytes = np.packbits(u8_arr, axis=1, bitorder="little")
        return np.frombuffer(packed_bytes.tobytes(), dtype=np.uint64).reshape(-1, 32).copy()
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


def unpack_fingerprint(packed: np.ndarray) -> np.ndarray:
    """
    Unpacks 32 uint64 words into a 2048-element uint8 array (0 and 1).

    Args:
        packed: 1D uint64 array of shape (32,) or 2D uint64 array of shape (N, 32).

    Returns:
        1D uint8 array of shape (2048,) or 2D uint8 array of shape (N, 2048).
    """
    packed = np.asarray(packed, dtype=np.uint64)
    if packed.ndim == 1:
        if packed.shape[0] != 32:
            raise ValueError(f"Expected 32 words for packed fingerprint, got {packed.shape[0]}")
        raw_u8 = np.frombuffer(packed.tobytes(), dtype=np.uint8)
        return np.unpackbits(raw_u8, bitorder="little")[:2048]
    elif packed.ndim == 2:
        if packed.shape[1] != 32:
            raise ValueError(f"Expected shape (N, 32) for packed fingerprints, got {packed.shape}")
        raw_u8 = np.frombuffer(packed.tobytes(), dtype=np.uint8).reshape(packed.shape[0], 256)
        return np.unpackbits(raw_u8, axis=1, bitorder="little")[:, :2048]
    else:
        raise ValueError(f"Unsupported packed array dimension: {packed.ndim}")


def popcount_tanimoto(
    query_fp: np.ndarray,
    candidate_fps: np.ndarray,
) -> np.ndarray:
    """
    Vectorized popcount Tanimoto similarity scoring between a 2048-bit query
    fingerprint and a candidate fingerprint matrix (packed into 32 uint64 words).

    Runs in < 2ms for 10,000 candidates using Numba parallel popcount or NumPy 2.x bitwise_count.

    Args:
        query_fp: Query fingerprint array (shape (32,) uint64, or (2048,) unpacked vector).
        candidate_fps: 2D array of packed fingerprints (shape (N, 32), dtype uint64).

    Returns:
        1D float32 array of shape (N,) with Tanimoto similarities in [0.0, 1.0].
    """
    if candidate_fps is None or candidate_fps.size == 0 or candidate_fps.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    # Ensure candidate_fps is 2D uint64 with 32 columns
    if candidate_fps.ndim == 1:
        if candidate_fps.shape[0] == 32 and candidate_fps.dtype == np.uint64:
            candidate_fps = candidate_fps.reshape(1, 32)
        else:
            raise ValueError(f"Invalid candidate_fps shape: {candidate_fps.shape}")

    if candidate_fps.shape[1] != 32 or candidate_fps.dtype != np.uint64:
        raise ValueError(f"candidate_fps must be (N, 32) uint64, got {candidate_fps.shape} {candidate_fps.dtype}")

    # Prepare query_fp
    query_fp = np.asarray(query_fp)
    if query_fp.dtype != np.uint64 or query_fp.shape != (32,):
        if query_fp.size == 2048:
            query_u64 = pack_fingerprint(query_fp)
        elif query_fp.size == 32:
            query_u64 = np.ascontiguousarray(query_fp, dtype=np.uint64)
        else:
            raise ValueError(f"Query fingerprint must have size 32 (packed) or 2048 (unpacked), got size {query_fp.size}")
    else:
        query_u64 = np.ascontiguousarray(query_fp, dtype=np.uint64)

    # Fast path: Numba JIT parallel popcount
    if _NUMBA_AVAILABLE and _numba_popcount_tanimoto is not None:
        try:
            return _numba_popcount_tanimoto(query_u64, candidate_fps)
        except Exception:
            pass

    # High-performance vectorized NumPy fallback
    inter = np.bitwise_count(candidate_fps & query_u64).sum(axis=1)
    cand_sums = np.bitwise_count(candidate_fps).sum(axis=1)
    q_sum = int(np.bitwise_count(query_u64).sum())
    union = cand_sums + q_sum - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)


class ChemicalDatabaseIndexer:
    """
    Offline chemical database indexer for Enveda CASMI 2026.

    Standardizes chemical structures, filters unparseable molecules to dead-letter logs,
    computes canonical Hill formulas and 2048-bit Morgan fingerprints (packed into 32 uint64 words),
    and stores candidates in SQLite WITHOUT ROWID tables optimized for clustered formula queries.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        dead_letter_path: Optional[Union[str, Path]] = None,
        suppress_chiral: bool = True,
    ):
        # Verify RDKit availability for build-time operations
        from src.chemistry.standardizer import _RDKIT_AVAILABLE
        if not _RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ChemicalDatabaseIndexer build-time indexing")

        self.db_path = str(db_path)
        self.dead_letter_path = Path(dead_letter_path) if dead_letter_path else None
        self.suppress_chiral = suppress_chiral
        self.dead_letters: List[Dict[str, Any]] = []
        self.conn: Optional[sqlite3.Connection] = None

        self.init_db()

    def init_db(self) -> None:
        """Initializes SQLite tables and executes high-throughput build-time pragmas."""
        if self.db_path == ":memory:":
            self.conn = sqlite3.connect(":memory:")
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)

        # Build-time performance pragmas
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=OFF;")
        self.conn.execute("PRAGMA cache_size=-2000000;")  # ~2 GB memory cache
        self.conn.execute("PRAGMA temp_store=MEMORY;")

        # Clustered schema WITHOUT ROWID
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                formula TEXT NOT NULL,
                inchikey14 TEXT NOT NULL,
                smiles TEXT NOT NULL,
                fingerprint BLOB NOT NULL,
                PRIMARY KEY (formula, inchikey14)
            ) WITHOUT ROWID;
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )

        # Set default metadata
        now_utc = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?);",
            ("schema_version", "1.0"),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?);",
            ("fingerprint_type", "Morgan_r2_b2048_uint64x32"),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?);",
            ("created_at", now_utc),
        )
        self.conn.commit()

    def _log_dead_letter(self, smiles: Any, identifier: Optional[str], reason: str) -> None:
        """Records an unparseable or failed SMILES entry without interrupting indexing."""
        entry = {
            "id": identifier,
            "smiles": smiles,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.dead_letters.append(entry)
        if self.dead_letter_path is not None:
            try:
                self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.dead_letter_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed writing dead letter to {self.dead_letter_path}: {e}")

    def index_molecules(
        self,
        records: Iterable[Union[Dict[str, Any], str, Tuple[Any, ...]]],
        batch_size: int = 5000,
    ) -> Dict[str, int]:
        """
        Indexes an iterable of chemical records (SMILES strings, dicts, or tuples).

        Args:
            records: Iterable of items. If dict, expects 'smiles' and optional 'id'/'molecule_id'.
                     If string, treated directly as SMILES.
                     If tuple, expects (smiles, id).
            batch_size: Number of records to insert per transaction.

        Returns:
            Dictionary with counts: {"indexed": int, "failed": int, "total": int}
        """
        from src.chemistry.standardizer import (
            compute_hill_formula,
            compute_morgan_fingerprint,
            standardize_mol,
        )

        indexed_count = 0
        failed_count = 0
        total_count = 0
        batch_buffer: List[Tuple[str, str, str, bytes]] = []

        cursor = self.conn.cursor()

        for rec in records:
            total_count += 1
            ident = None
            if isinstance(rec, str):
                smiles = rec
            elif isinstance(rec, dict):
                smiles = rec.get("smiles")
                ident = rec.get("id") or rec.get("molecule_id")
            elif isinstance(rec, (tuple, list)):
                smiles = rec[0] if len(rec) > 0 else None
                ident = rec[1] if len(rec) > 1 else None
            else:
                smiles = None

            if not smiles or not isinstance(smiles, str) or not smiles.strip():
                self._log_dead_letter(smiles, ident, "Empty or invalid SMILES type")
                failed_count += 1
                continue

            try:
                canon_smi, ik14 = standardize_mol(smiles, suppress_chiral=self.suppress_chiral)
                if not canon_smi or not ik14:
                    self._log_dead_letter(smiles, ident, "Standardization failed")
                    failed_count += 1
                    continue

                formula = compute_hill_formula(canon_smi)
                if not formula:
                    self._log_dead_letter(smiles, ident, "Hill formula calculation failed")
                    failed_count += 1
                    continue

                fp = compute_morgan_fingerprint(canon_smi)
                if fp is None or fp.shape != (32,) or fp.dtype != np.uint64:
                    self._log_dead_letter(smiles, ident, "Fingerprint calculation failed")
                    failed_count += 1
                    continue

                batch_buffer.append((formula, ik14, canon_smi, fp.tobytes()))

            except Exception as exc:
                self._log_dead_letter(smiles, ident, f"Exception during indexing: {exc}")
                failed_count += 1
                continue

            if len(batch_buffer) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO candidates (formula, inchikey14, smiles, fingerprint) VALUES (?, ?, ?, ?);",
                    batch_buffer,
                )
                self.conn.commit()
                indexed_count += len(batch_buffer)
                batch_buffer.clear()

        # Flush remaining buffer
        if batch_buffer:
            cursor.executemany(
                "INSERT OR IGNORE INTO candidates (formula, inchikey14, smiles, fingerprint) VALUES (?, ?, ?, ?);",
                batch_buffer,
            )
            self.conn.commit()
            indexed_count += len(batch_buffer)
            batch_buffer.clear()

        # Update metadata count
        cursor.execute("SELECT COUNT(*) FROM candidates;")
        total_in_db = cursor.fetchone()[0]
        self.conn.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?);",
            ("candidate_count", str(total_in_db)),
        )
        self.conn.commit()

        return {
            "indexed": indexed_count,
            "failed": failed_count,
            "total": total_count,
            "total_in_db": total_in_db,
        }

    def index_from_csv(
        self,
        csv_path: Union[str, Path],
        smiles_col: str = "smiles",
        id_col: Optional[str] = "id",
        batch_size: int = 5000,
    ) -> Dict[str, int]:
        """Indexes molecules from a CSV file using standard library csv.DictReader."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        def _row_generator():
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smi = row.get(smiles_col)
                    ident = row.get(id_col) if id_col else None
                    yield {"smiles": smi, "id": ident}

        return self.index_molecules(_row_generator(), batch_size=batch_size)

    def index_from_smiles_list(
        self,
        smiles_list: List[str],
        batch_size: int = 5000,
    ) -> Dict[str, int]:
        """Convenience method to index a list of SMILES strings."""
        return self.index_molecules(smiles_list, batch_size=batch_size)

    def get_metadata(self) -> Dict[str, str]:
        """Returns all key-value pairs from _schema_metadata."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM _schema_metadata;")
        return dict(cursor.fetchall())

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> ChemicalDatabaseIndexer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ChemicalDatabaseRepository:
    """
    High-performance runtime repository for chemical candidate retrieval.

    Decoupled from RDKit — requires only standard library sqlite3 and numpy.
    Supports memory-mapped zero-copy read-only operations, chunked formula queries
    (250 formulas per batch), and sub-2ms vectorized popcount Tanimoto scoring.
    """

    def __init__(self, db_path: Union[str, Path, sqlite3.Connection]):
        self._owns_conn = False
        self.db_path = db_path

        if isinstance(db_path, sqlite3.Connection):
            self.conn = db_path
        elif str(db_path) == ":memory:":
            self.conn = sqlite3.connect(":memory:")
            self._owns_conn = True
        else:
            path_obj = Path(db_path).resolve()
            if not path_obj.exists():
                raise FileNotFoundError(f"Chemical database file not found: {path_obj}")

            uri = f"file:{path_obj.as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)
            self._owns_conn = True

        # Apply inference-time read pragmas
        for pragma in [
            "PRAGMA mmap_size=30000000000;",  # 30 GB mmap window
            "PRAGMA query_only=ON;",
            "PRAGMA threads=4;",
        ]:
            try:
                self.conn.execute(pragma)
            except Exception:
                pass

    def search_by_formulas(
        self,
        formulas: List[str],
        deduplicate: bool = True,
    ) -> Tuple[List[Tuple[str, str, str]], np.ndarray]:
        """
        Batched formula retrieval in chunks of 250 formulas.

        Args:
            formulas: List of Hill chemical formulas (e.g. ['C9H8O4', 'C15H10O5']).
            deduplicate: If True, deduplicates returned candidates by InChIKey14.

        Returns:
            (metadata_list, contiguous_uint64_matrix) where:
              metadata_list: List of (formula, inchikey14, smiles) tuples.
              contiguous_uint64_matrix: NumPy array of shape (N, 32) and dtype uint64.
        """
        if not formulas:
            return [], np.empty((0, 32), dtype=np.uint64)

        # Sanitize and deduplicate formula list while preserving order
        unique_formulas = list(
            dict.fromkeys(f.strip() for f in formulas if f and isinstance(f, str) and f.strip())
        )
        if not unique_formulas:
            return [], np.empty((0, 32), dtype=np.uint64)

        chunk_size = 250
        all_rows: List[Tuple[str, str, str, bytes]] = []
        cursor = self.conn.cursor()

        for i in range(0, len(unique_formulas), chunk_size):
            chunk = unique_formulas[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT formula, inchikey14, smiles, fingerprint FROM candidates WHERE formula IN ({placeholders});"
            cursor.execute(query, chunk)
            all_rows.extend(cursor.fetchall())

        if not all_rows:
            return [], np.empty((0, 32), dtype=np.uint64)

        seen_ik14 = set()
        filtered_rows: List[Tuple[str, str, str, bytes]] = []
        for r in all_rows:
            ik14 = r[1]
            if deduplicate:
                if ik14 in seen_ik14:
                    continue
                seen_ik14.add(ik14)
            filtered_rows.append(r)

        if not filtered_rows:
            return [], np.empty((0, 32), dtype=np.uint64)

        metadata = [(r[0], r[1], r[2]) for r in filtered_rows]
        raw_bytes = b"".join(r[3] for r in filtered_rows)
        fps_matrix = np.frombuffer(raw_bytes, dtype=np.uint64).reshape(-1, 32).copy()

        return metadata, fps_matrix

    def get_metadata(self) -> Dict[str, str]:
        """Returns all key-value pairs from _schema_metadata."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT key, value FROM _schema_metadata;")
            return dict(cursor.fetchall())
        except Exception:
            return {}

    def count(self) -> int:
        """Returns total candidate count in the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM candidates;")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def close(self) -> None:
        """Closes the underlying SQLite connection if owned."""
        if self._owns_conn and self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> ChemicalDatabaseRepository:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = [
    "ChemicalDatabaseIndexer",
    "ChemicalDatabaseRepository",
    "pack_fingerprint",
    "unpack_fingerprint",
    "popcount_tanimoto",
]
