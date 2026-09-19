from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class NetworkCandidate:
    source_id: str
    scaffold_smiles: str
    inchikey14: str
    transformation: str
    network_score: float
    source_tier: str = "track_network"


class TransductiveMolecularNetwork:
    """Module 3: Neutral-mass normalized transductive test-set molecular network."""

    DELTA_LIBRARY = {
        176.0321: "+Glucuronide",
        162.0528: "+Hexose",
        146.0579: "+Rhamnose",
        132.0423: "+Pentose",
        86.0004: "+Malonyl",
        42.0106: "+Acetyl",
        14.0156: "+Methyl",
    }

    def __init__(self, cosine_threshold: float = 0.7, mass_tolerance_ppm: float = 15.0):
        self.cosine_threshold = cosine_threshold
        self.mass_tolerance_ppm = mass_tolerance_ppm
        self.spectra: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}

    def add_spectrum(
        self, spec_id: str, neutral_mass: float, embedding: np.ndarray, peaks: np.ndarray
    ) -> None:
        """Register a spectrum in the transductive network with normalized embedding."""
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        norm_emb = emb / (norm + 1e-12) if norm > 0 else emb
        self.spectra[spec_id] = (
            float(neutral_mass),
            norm_emb,
            np.asarray(peaks, dtype=np.float64),
        )

    def match_delta(self, delta_m: float) -> Optional[str]:
        """Match observed mass delta against botanical library within mass tolerance (ppm)."""
        for delta_ref, name in self.DELTA_LIBRARY.items():
            err_ppm = abs(delta_m - delta_ref) / delta_ref * 1e6
            if err_ppm <= self.mass_tolerance_ppm:
                return name
        return None

    def count_shared_peaks(
        self, peaks1: np.ndarray, peaks2: np.ndarray, mz_tolerance: float = 0.02
    ) -> int:
        """Count shared fragment peaks within mass tolerance."""
        if len(peaks1) == 0 or len(peaks2) == 0:
            return 0
        p1_mz = peaks1[:, 0] if peaks1.ndim == 2 else peaks1
        p2_mz = peaks2[:, 0] if peaks2.ndim == 2 else peaks2
        shared = 0
        for p1 in p1_mz:
            if np.any(np.abs(p2_mz - p1) <= mz_tolerance):
                shared += 1
        return shared

    def propagate_scaffold(
        self, target_id: str, known_scaffolds: Dict[str, Tuple[str, str]]
    ) -> List[NetworkCandidate]:
        """Propagate high-confidence solved scaffolds to related targets in the test network."""
        if target_id not in self.spectra:
            return []

        target_m, target_emb, target_peaks = self.spectra[target_id]
        results: List[NetworkCandidate] = []

        for source_id, (smiles, ik14) in known_scaffolds.items():
            if source_id == target_id or source_id not in self.spectra:
                continue
            source_m, source_emb, source_peaks = self.spectra[source_id]
            cos_sim = float(np.dot(target_emb, source_emb))
            if cos_sim < self.cosine_threshold:
                continue

            delta = abs(target_m - source_m)
            trans_name = self.match_delta(delta)
            if trans_name is not None:
                # Require >= 2 shared fragment peaks for transductive co-validation
                shared_peaks = self.count_shared_peaks(target_peaks, source_peaks)
                if shared_peaks >= 2:
                    results.append(
                        NetworkCandidate(
                            source_id=source_id,
                            scaffold_smiles=smiles,
                            inchikey14=ik14,
                            transformation=trans_name,
                            network_score=cos_sim,
                        )
                    )

        # Sort candidates descending by network score (cosine similarity)
        results.sort(key=lambda c: c.network_score, reverse=True)
        return results
