from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class NetworkCandidate:
    source_id: str
    scaffold_smiles: str
    inchikey14: str
    transformation: str
    network_score: float
    source_tier: str = "track_network"


# 16 Verified Plant Natural Product Transformations across 4 classes
BOTANICAL_TRANSFORMATIONS: Dict[str, float] = {
    # Glycosylations
    "Hexose": 162.0528,
    "Pentose": 132.0423,
    "Rhamnose": 146.0579,  # Rhamnose / Deoxyhexose
    "Glucuronide": 176.0321,
    # Acylations & Alkylations
    "Methyl": 14.0157,
    "Acetyl": 42.0106,
    "Malonyl": 86.0004,
    "Prenyl": 68.0626,
    # Phenylpropanoids
    "Galloyl": 152.0110,
    "Caffeoyl": 162.0317,
    "Feruloyl": 176.0473,
    "Coumaroyl": 146.0368,
    "Sinapoyl": 206.0579,
    # Functionalizations
    "Hydroxylation": 15.9949,
    "Sulfate": 79.9568,
    "Phosphate": 79.9663,
}

class _DeltaLibrary(dict):
    """Dictionary mapping neutral mass deltas to transformation names with legacy alias support."""

    _ALIASES = {
        14.0156: "+Methyl",
    }

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, (int, float)):
            for k in self._ALIASES:
                if abs(key - k) < 1e-6:
                    return True
        return False

    def __getitem__(self, key: object) -> str:
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, (int, float)):
                for k, v in self._ALIASES.items():
                    if abs(key - k) < 1e-6:
                        return v
            raise

    def get(self, key: object, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


DELTA_LIBRARY: Dict[float, str] = _DeltaLibrary({
    # Glycosylations
    162.0528: "+Hexose",
    132.0423: "+Pentose",
    146.0579: "+Rhamnose",
    176.0321: "+Glucuronide",
    # Acylations & Alkylations
    14.0157: "+Methyl",
    42.0106: "+Acetyl",
    86.0004: "+Malonyl",
    68.0626: "+Prenyl",
    # Phenylpropanoids
    152.0110: "+Galloyl",
    162.0317: "+Caffeoyl",
    176.0473: "+Feruloyl",
    146.0368: "+Coumaroyl",
    206.0579: "+Sinapoyl",
    # Functionalizations
    15.9949: "+Hydroxylation",
    79.9568: "+Sulfate",
    79.9663: "+Phosphate",
})


class TransductiveMolecularNetwork:
    """Module 3: Neutral-mass normalized transductive test-set molecular network with botanical knowledge base."""

    BOTANICAL_TRANSFORMATIONS = BOTANICAL_TRANSFORMATIONS
    DELTA_LIBRARY = DELTA_LIBRARY

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

    def match_delta(
        self, delta_m: float, tolerance_ppm: Optional[float] = None
    ) -> Optional[str]:
        """Match observed mass delta against botanical library within mass tolerance (ppm).

        Supports bidirectional matching: positive deltas return '+<Name>',
        negative deltas return '-<Name>'.
        """
        tol = self.mass_tolerance_ppm if tolerance_ppm is None else tolerance_ppm
        delta_abs = abs(delta_m)
        sign = "-" if delta_m < 0 else "+"

        best_match: Optional[str] = None
        min_err = float("inf")

        for delta_ref, name in self.DELTA_LIBRARY.items():
            err_ppm = abs(delta_abs - delta_ref) / delta_ref * 1e6
            if err_ppm <= tol and err_ppm < min_err:
                min_err = err_ppm
                base_name = name.lstrip("+-")
                best_match = f"{sign}{base_name}"

        return best_match

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
            if np.any(np.abs(p2_mz - p1) <= (mz_tolerance + 1e-9)):
                shared += 1
        return shared

    def propagate_scaffold(
        self,
        target_id: str,
        known_scaffolds: Dict[str, Tuple[str, str]],
        min_shared_peaks: int = 2,
        directional: bool = False,
        tolerance_ppm: Optional[float] = None,
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

            delta = (target_m - source_m) if directional else abs(target_m - source_m)
            trans_name = self.match_delta(delta, tolerance_ppm=tolerance_ppm)
            if trans_name is not None:
                # Require >= min_shared_peaks shared fragment peaks for transductive co-validation
                shared_peaks = self.count_shared_peaks(target_peaks, source_peaks)
                if shared_peaks >= min_shared_peaks:
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
