from dataclasses import dataclass
from typing import Dict, List, Tuple
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

    def __init__(self):
        self.calibration_table: Dict[Tuple[str, str], float] = {
            ("timsTOF", "timsTOF"): 0.88,
            ("timsTOF", "Orbitrap"): 0.82,
            ("timsTOF", "QTOF"): 0.80,
            ("generic", "generic"): 0.85,
        }

    def get_calibration_threshold(self, query_inst: str, lib_inst: str) -> float:
        return self.calibration_table.get(
            (query_inst, lib_inst), self.calibration_table[("generic", "generic")]
        )

    def evaluate_candidates(
        self,
        query_emb: np.ndarray,
        raw_candidates: List[Tuple[str, str, np.ndarray, str]],
        query_instrument: str = "timsTOF",
    ) -> Tuple[List[RetrievedCandidate], bool]:
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
