from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class RetrievedCandidate:
    smiles: str
    inchikey14: str
    score: float
    instrument: str
    source_tier: str = "track1_dreams"


class CalibratedDreaMSRetriever:
    """Track 1: Instrument-stratified cosine thresholding and FP16 re-scoring."""

    def __init__(
        self,
        index: Optional[Union[str, Path, Any]] = None,
        calibration_table: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        if calibration_table is not None:
            self.calibration_table: Dict[Tuple[str, str], float] = dict(
                calibration_table
            )
        else:
            self.calibration_table = {
                ("timsTOF", "timsTOF"): 0.88,
                ("timsTOF", "Orbitrap"): 0.82,
                ("timsTOF", "QTOF"): 0.80,
                ("generic", "generic"): 0.85,
            }

        self.index = None
        if index is not None:
            if hasattr(index, "search"):
                self.index = index
            else:
                from src.data.spectral_indexer import SpectralIndex

                self.index = SpectralIndex(index)

    @property
    def has_index(self) -> bool:
        """Check if an underlying spectral vector index is attached."""
        return self.index is not None

    def get_calibration_threshold(self, query_inst: str, lib_inst: str) -> float:
        return self.calibration_table.get(
            (query_inst, lib_inst),
            self.calibration_table.get(("generic", "generic"), 0.85),
        )

    def retrieve(
        self,
        query_emb: np.ndarray,
        precursor_mz: float,
        polarity: str = "positive",
        query_instrument: str = "timsTOF",
        tolerance_ppm: float = 10.0,
        top_k: int = 25,
        **kwargs: Any,
    ) -> Tuple[List[RetrievedCandidate], bool]:
        """Query the reference SpectralIndex and apply instrument-stratified calibration."""
        if self.index is None:
            return [], False

        tol_ppm = float(kwargs.get("ppm_tolerance", tolerance_ppm))

        hits = self.index.search(
            query_emb=query_emb,
            precursor_mz=precursor_mz,
            polarity=polarity,
            ppm_tolerance=tol_ppm,
            top_k=top_k,
            **kwargs,
        )

        if not hits:
            return [], False

        scored: List[RetrievedCandidate] = []
        is_locked_rank1 = False

        for hit in hits:
            score = float(hit.cosine_score)
            threshold = self.get_calibration_threshold(query_instrument, hit.instrument)
            if score >= threshold and not is_locked_rank1:
                is_locked_rank1 = True

            scored.append(
                RetrievedCandidate(
                    smiles=hit.smiles,
                    inchikey14=hit.inchikey14,
                    score=score,
                    instrument=hit.instrument,
                    source_tier="track1_dreams",
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored, is_locked_rank1

    def evaluate_candidates(
        self,
        query_emb: np.ndarray,
        raw_candidates: Optional[List[Tuple[str, str, np.ndarray, str]]] = None,
        query_instrument: str = "timsTOF",
        precursor_mz: Optional[float] = None,
        polarity: str = "positive",
        tolerance_ppm: float = 10.0,
        top_k: int = 25,
        **kwargs: Any,
    ) -> Tuple[List[RetrievedCandidate], bool]:
        """Evaluate candidate spectra against query embedding.

        Maintains 100% backward compatibility when raw_candidates is provided.
        When raw_candidates is None and an index is injected, delegates to self.retrieve.
        """
        if raw_candidates is not None:
            if not raw_candidates:
                return [], False

            query_fp16 = query_emb.astype(np.float16)
            query_norm = float(np.linalg.norm(query_fp16))
            scored: List[RetrievedCandidate] = []
            is_locked_rank1 = False

            for smiles, ik14, lib_emb, lib_inst in raw_candidates:
                lib_fp16 = lib_emb.astype(np.float16)
                lib_norm = float(np.linalg.norm(lib_fp16))
                denom = query_norm * lib_norm + 1e-12
                dot_prod = float(np.dot(query_fp16, lib_fp16))
                cos_sim = dot_prod / denom if denom > 1e-12 else 0.0
                if np.isnan(cos_sim):
                    cos_sim = 0.0
                threshold = self.get_calibration_threshold(query_instrument, lib_inst)

                if cos_sim >= threshold and not is_locked_rank1:
                    is_locked_rank1 = True
                scored.append(
                    RetrievedCandidate(
                        smiles=smiles,
                        inchikey14=ik14,
                        score=cos_sim,
                        instrument=lib_inst,
                    )
                )

            scored.sort(key=lambda x: x.score, reverse=True)
            return scored, is_locked_rank1

        if self.index is not None and precursor_mz is not None:
            return self.retrieve(
                query_emb=query_emb,
                precursor_mz=precursor_mz,
                polarity=polarity,
                query_instrument=query_instrument,
                tolerance_ppm=tolerance_ppm,
                top_k=top_k,
                **kwargs,
            )

        return [], False

    def close(self) -> None:
        """Safely close underlying spectral index file handles."""
        if self.index is not None and hasattr(self.index, "close"):
            self.index.close()

    def __enter__(self) -> CalibratedDreaMSRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

