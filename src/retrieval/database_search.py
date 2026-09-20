from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class DBCandidate:
    smiles: str
    inchikey14: str
    tanimoto_score: float
    source_tier: str = "track2_db"


class SoftDatabaseSearcher:
    """Track 2: Soft formula-sliced candidate retrieval with vectorized Tanimoto scoring & fallback."""

    def __init__(
        self,
        db: Optional[Union[Dict[str, List[Tuple[str, str, np.ndarray]]], str, Path, Any]] = None,
        fallback_scaffolds: Optional[List[Tuple[str, str, np.ndarray]]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.fallback_scaffolds = fallback_scaffolds or []
        self.repo: Optional[Any] = None
        self.db: Dict[str, List[Tuple[str, str, np.ndarray]]] = {}

        target_db_path = db_path
        if target_db_path is None and isinstance(db, (str, Path)):
            target_db_path = db

        if target_db_path is not None:
            from src.data.db_indexer import ChemicalDatabaseRepository
            self.repo = ChemicalDatabaseRepository(target_db_path)
        elif hasattr(db, "search_by_formulas"):
            self.repo = db
        elif isinstance(db, dict):
            self.db = db
        else:
            self.db = {}

    def batch_tanimoto(self, query_fp: np.ndarray, db_fps: np.ndarray) -> np.ndarray:
        """Vectorized Tanimoto calculation over 2D candidate fingerprint matrix."""
        if db_fps.shape[0] == 0:
            return np.array([], dtype=np.float32)
        q = query_fp.astype(np.float32)
        fps = db_fps.astype(np.float32)
        intersection = np.dot(fps, q)
        query_sum = np.sum(q)
        db_sums = np.sum(fps, axis=1)
        union = db_sums + query_sum - intersection
        return np.where(union > 0, intersection / (union + 1e-12), 0.0).astype(np.float32)

    def tanimoto(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Pairwise Tanimoto similarity."""
        intersection = float(np.sum(np.logical_and(fp1, fp2)))
        union = float(np.sum(np.logical_or(fp1, fp2)))
        if union <= 0:
            return 0.0
        return float(intersection / (union + 1e-12))

    def search_formulas(
        self, formulas: List[str], query_fp: np.ndarray
    ) -> List[DBCandidate]:
        if self.repo is not None:
            from src.data.db_indexer import popcount_tanimoto

            metadata, fps_matrix = self.repo.search_by_formulas(formulas, deduplicate=True)
            candidates: List[DBCandidate] = []

            if len(metadata) > 0 and fps_matrix.size > 0:
                scores = popcount_tanimoto(query_fp, fps_matrix)
                for (form, ik14, smiles), score in zip(metadata, scores):
                    candidates.append(
                        DBCandidate(
                            smiles=smiles,
                            inchikey14=ik14,
                            tanimoto_score=float(score),
                            source_tier="track2_db",
                        )
                    )

            # Fallback if zero candidates found
            if not candidates:
                for smiles, ik14, mol_fp in self.fallback_scaffolds[:25]:
                    candidates.append(
                        DBCandidate(
                            smiles=smiles,
                            inchikey14=ik14,
                            tanimoto_score=0.01,
                            source_tier="track2_fallback",
                        )
                    )

            candidates.sort(key=lambda x: x.tanimoto_score, reverse=True)
            return candidates

        # In-memory dictionary retrieval (100% backward compatible)
        candidates: List[DBCandidate] = []
        seen_ik14 = set()

        batch_mols: List[Tuple[str, str]] = []
        batch_fps: List[np.ndarray] = []

        for form in formulas:
            for smiles, ik14, mol_fp in self.db.get(form, []):
                if ik14 in seen_ik14:
                    continue
                seen_ik14.add(ik14)
                batch_mols.append((smiles, ik14))
                batch_fps.append(mol_fp)

        if batch_fps:
            fps_matrix = np.stack(batch_fps, axis=0)
            scores = self.batch_tanimoto(query_fp, fps_matrix)
            for (smiles, ik14), score in zip(batch_mols, scores):
                candidates.append(
                    DBCandidate(
                        smiles=smiles,
                        inchikey14=ik14,
                        tanimoto_score=float(score),
                    )
                )

        # Fallback if zero candidates found
        if not candidates:
            for smiles, ik14, mol_fp in self.fallback_scaffolds[:25]:
                candidates.append(
                    DBCandidate(
                        smiles=smiles,
                        inchikey14=ik14,
                        tanimoto_score=0.01,
                        source_tier="track2_fallback",
                    )
                )

        candidates.sort(key=lambda x: x.tanimoto_score, reverse=True)
        return candidates

    def close(self) -> None:
        """Closes the underlying repository connection if active."""
        if self.repo is not None and hasattr(self.repo, "close"):
            self.repo.close()

    def __enter__(self) -> SoftDatabaseSearcher:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

