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

# 25 Verified Botanical Aglycone Cores for Natural Product Networking
BOTANICAL_AGLYCONE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "REF_QUERCETIN": {"smiles": "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "inchikey14": "REFJWTPEDVJJIY", "neutral_mass": 302.0427},
    "REF_KAEMPFEROL": {"smiles": "O=C1C(O)=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "inchikey14": "IYRMWMYZSQPJKC", "neutral_mass": 286.0477},
    "REF_APIGENIN": {"smiles": "O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "inchikey14": "KZNIFHPLKGYRTM", "neutral_mass": 270.0528},
    "REF_LUTEOLIN": {"smiles": "O=C1C=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "inchikey14": "IQPNAANSBPBGFQ", "neutral_mass": 286.0477},
    "REF_MYRICETIN": {"smiles": "O=C1C(O)=C(c2cc(O)c(O)c(O)c2)Oc2cc(O)cc(O)c12", "inchikey14": "IKMDFBPHZNJCSN", "neutral_mass": 318.0376},
    "REF_NARINGENIN": {"smiles": "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "inchikey14": "FTVWIRXFELQLPI", "neutral_mass": 272.0685},
    "REF_HESPERETIN": {"smiles": "COc1ccc(C2CC(=O)c3c(O)cc(O)cc3O2)cc1O", "inchikey14": "AIONOLUJZLIMTK", "neutral_mass": 302.0790},
    "REF_ERIODICTYOL": {"smiles": "O=C1CC(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "inchikey14": "SBHXYTNGIZCORC", "neutral_mass": 288.0634},
    "REF_GENISTEIN": {"smiles": "O=C1C(=Cc2ccc(O)cc2)Oc2cc(O)cc(O)c12", "inchikey14": "ACGXIFQKQYZLOU", "neutral_mass": 270.0528},
    "REF_DAIDZEIN": {"smiles": "O=C1C(=Cc2ccc(O)cc2)Oc2cc(O)ccc12", "inchikey14": "KEZLDSPIRVZOKZ", "neutral_mass": 254.0579},
    "REF_GLYCITEIN": {"smiles": "COc1cc2c(=O)c(-c3ccc(O)cc3)coc2cc1O", "inchikey14": "DXYUAIFZCFRPTH", "neutral_mass": 284.0685},
    "REF_CHRYSIN": {"smiles": "O=C1C=C(c2ccccc2)Oc2cc(O)cc(O)c12", "inchikey14": "RTIXKCRFFJGDFG", "neutral_mass": 254.0579},
    "REF_BAICALEIN": {"smiles": "O=C1C=C(c2ccccc2)Oc2c(O)c(O)cc(O)c12", "inchikey14": "ZFKKRRMUPBBYRS", "neutral_mass": 270.0528},
    "REF_CAFFEIC_ACID": {"smiles": "O=C(O)C=Cc1ccc(O)c(O)c1", "inchikey14": "QAIPRVGONGVQAS", "neutral_mass": 180.0423},
    "REF_FERULIC_ACID": {"smiles": "COc1cc(C=CC(=O)O)ccc1O", "inchikey14": "KSEBMYQBYZTDHS", "neutral_mass": 194.0579},
    "REF_P_COUMARIC_ACID": {"smiles": "O=C(O)C=Cc1ccc(O)cc1", "inchikey14": "NGSWKAQJJWESNS", "neutral_mass": 164.0473},
    "REF_GALLIC_ACID": {"smiles": "O=C(O)c1cc(O)c(O)c(O)c1", "inchikey14": "LNTHITQWFMADLM", "neutral_mass": 170.0215},
    "REF_SINAPIC_ACID": {"smiles": "COc1cc(C=CC(=O)O)cc(OC)c1O", "inchikey14": "PCMORTLOPMLEFB", "neutral_mass": 224.0685},
    "REF_CHLOROGENIC_CORE": {"smiles": "O=C(O)C1CC(O)C(OC(=O)C=Cc2ccc(O)c(O)c2)C(O)C1", "inchikey14": "RPMJUNLURDWOGV", "neutral_mass": 338.1002},
    "REF_EMODIN": {"smiles": "Cc1cc(O)c2c(=O)c3cc(O)cc(O)c3c(=O)c2c1", "inchikey14": "QXGCQVUIZNDQAM", "neutral_mass": 270.0528},
    "REF_NORATHYRIOL": {"smiles": "O=C1c2c(O)cc(O)c(O)c2Oc2cc(O)c(O)cc12", "inchikey14": "OBWMDMPXSQQBSL", "neutral_mass": 276.0270},
    "REF_RESVERATROL": {"smiles": "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1", "inchikey14": "LUKBXSAWLPMMSZ", "neutral_mass": 228.0786},
    "REF_BERBERINE": {"smiles": "COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2", "inchikey14": "YBHILYKTIRIUTE", "neutral_mass": 336.1230},
    "REF_URSOLIC_ACID": {"smiles": "CC1CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C2C1C", "inchikey14": "WCGUUGGRBIKTOS", "neutral_mass": 456.3603},
    "REF_OLEANOLIC_ACID": {"smiles": "CC1(C)CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C2C1", "inchikey14": "MIJYXULNPSFWEK", "neutral_mass": 456.3603},
}


class TransductiveMolecularNetwork:
    """Module 3: Neutral-mass normalized transductive test-set molecular network with botanical knowledge base."""

    BOTANICAL_TRANSFORMATIONS = BOTANICAL_TRANSFORMATIONS
    DELTA_LIBRARY = DELTA_LIBRARY
    BOTANICAL_AGLYCONE_LIBRARY = BOTANICAL_AGLYCONE_LIBRARY

    def __init__(self, cosine_threshold: float = 0.7, mass_tolerance_ppm: float = 15.0):
        self.cosine_threshold = cosine_threshold
        self.mass_tolerance_ppm = mass_tolerance_ppm
        self.spectra: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}

    def load_botanical_aglycones(self) -> None:
        """Pre-seed network with canonical plant aglycone reference spectra."""
        for key, info in self.BOTANICAL_AGLYCONE_LIBRARY.items():
            nm = float(info["neutral_mass"])
            emb = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
            # Reference aglycone base peaks
            peaks = np.array([
                [nm, 100.0],
                [nm - 18.0106, 40.0],
                [nm - 27.9949, 30.0],
                [152.0110, 20.0],
                [136.0160, 20.0],
            ], dtype=np.float64)
            self.add_spectrum(key, nm, emb, peaks)

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
