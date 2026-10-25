"""
Botanical substructure and diagnostic neutral-loss prior scoring module.

Provides:
- BotanicalSubstructureScorer: Identifies observed diagnostic botanical neutral losses
  (hexose, rhamnose, pentose, glucuronide, galloyl, caffeoyl, depsidone/lactone) and
  retro-Diels-Alder (RDA) fragment ions from fused MS/MS spectra, and evaluates candidate
  molecules using precision SMARTS substructure matching.
- Penalizes synthetic sulfonamides and halogenated rings when the query spectrum exhibits
  pure natural product fragmentation patterns.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    _RDKIT_AVAILABLE = False


def extract_peak_mzs(fused_peaks: Any) -> List[float]:
    """
    Resiliently extract a list of peak m/z values from varied MS/MS peak inputs.

    Supports:
        - 2D np.ndarray of shape (N, 2)
        - 1D np.ndarray of m/z values
        - List/tuple of (m/z, intensity) pairs or numbers
        - Delimited string or JSON peak representations
    """
    if fused_peaks is None:
        return []

    if isinstance(fused_peaks, np.ndarray):
        if fused_peaks.ndim == 2 and fused_peaks.shape[1] >= 1:
            return [float(x) for x in fused_peaks[:, 0] if np.isfinite(x)]
        elif fused_peaks.ndim == 1:
            return [float(x) for x in fused_peaks if np.isfinite(x)]
        return []

    if isinstance(fused_peaks, (list, tuple)):
        mzs: List[float] = []
        for p in fused_peaks:
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 1:
                try:
                    val = float(p[0])
                    if np.isfinite(val):
                        mzs.append(val)
                except (ValueError, TypeError):
                    pass
            elif isinstance(p, (int, float, np.number)):
                try:
                    val = float(p)
                    if np.isfinite(val):
                        mzs.append(val)
                except (ValueError, TypeError):
                    pass
        return mzs

    if isinstance(fused_peaks, str):
        try:
            from src.reranking.fragment_library import parse_peaks_field
            parsed = parse_peaks_field(fused_peaks)
            return [m for m, _ in parsed]
        except Exception:
            return []

    return []


# Precision botanical SMARTS definitions
BOTANICAL_SMARTS_DEFINITIONS: Dict[str, List[str]] = {
    # 1. Hexose loss: dehydrated hexopyranose (glucose, galactose, mannose)
    "hexose": [
        "[OX2,OX1-][CH2][CH,C]1O[CH,C][CH,C]([O,o])[CH,C]([O,o])[CH,C]1[O,o]",
        "O1C(CO)C(O)C(O)C(O)C1",
    ],
    # 2. Rhamnose / Deoxyhexose: 6-deoxyhexopyranose (methyl group attached to pyranose ring)
    "rhamnose": [
        "[CH3][CH,C]1O[CH,C][CH,C]([O,o])[CH,C]([O,o])[CH,C]1[O,o]",
    ],
    # 3. Pentose loss: 5-carbon sugar (arabinose, xylose: pentopyranose or pentofuranose)
    "pentose": [
        "[OX2]1[CH2X4][CH]([OX2])[CH]([OX2])[CH]([OX2])[CH]1",
        "[OX2H,OX1-][CH2][CH]1[OX2][CH][CH]([OX2])[CH]1[OX2]",
        "C1OCC(O)C(O)C1O",
        "OCC1OC(O)C(O)C1",
    ],
    # 4. Glucuronide loss: glucuronic acid (pyranose with carboxylic acid on C6)
    "glucuronide": [
        "[CX3](=[OX1])([OX2H,OX1-])[CH,C]1O[CH,C][CH,C]([O,o])[CH,C]([O,o])[CH,C]1[O,o]",
        "O=C(O)C1OC(O)C(O)C(O)C1",
    ],
    # 5. Galloyl loss / fragment: 3,4,5-trihydroxybenzoyl group (gallic acid ester)
    "galloyl": [
        "c1cc(O)c(O)c(O)c1",
        "[CX3](=O)c1cc(O)c(O)c(O)c1",
        "c1c(O)c(O)c(O)cc1",
    ],
    # 6. Caffeoyl loss: (E)-3-(3,4-dihydroxyphenyl)acryloyl (caffeic acid ester)
    "caffeoyl": [
        "c1cc(O)c(O)cc1C=CC(=O)",
        "[OX2H,OX1-][c]1[c]([OX2H,OX1-])[c][c]([c][c]1)[CH]=[CH][CX3](=O)",
    ],
    # 7. Depsidone / lactone ester: dibenzo[b,e][1,4]dioxepin-11-one, aryl esters, lactones
    "depsidone": [
        "O=C1Oc2ccccc2Oc2ccccc12",
        "c1ccc2c(c1)OC(=O)c1ccccc1O2",
        "[c]1[c][c][c]2[c]1[OX2][CX3](=O)[c]1[c][c][c][c][c]1[OX2]2",
        "c1ccccc1C(=O)Oc2ccccc2",
    ],
    # 8. Flavonoid / chromone core: chromone, flavone, flavonol, flavanone, isoflavone, flavan
    "flavonoid": [
        "O=c1ccoc2ccccc12",
        "O=c1c[c,C]oc2ccccc12",
        "O=C1[#6][#6]Oc2ccccc12",
        "[#6]1~[#6]~[#6](-[c]2ccccc2)~[OX2]~[c]3ccccc13",
    ],
}

# Synthetic distractor patterns to penalize when botanical fragmentation is observed
SYNTHETIC_DISTRACTOR_SMARTS: Dict[str, List[str]] = {
    "sulfonamide": [
        "S(=O)(=O)N",
        "[#16X4](=[OX1])(=[OX1])[#7]",
    ],
    "halogenated_ring": [
        "c[F,Cl,Br,I]",
    ],
}


class BotanicalSubstructureScorer:
    """
    Substructure and neutral-loss scorer prioritizing natural products and botanical classes.

    Identifies observed diagnostic neutral losses and retro-Diels-Alder (RDA) fragments,
    matches them to candidate SMARTS patterns, and severely downweights synthetic distractors.
    """

    def __init__(self, tolerance_da: float = 0.025):
        self.tolerance_da = tolerance_da

        # Precompile SMARTS patterns if RDKit is available
        self._botanical_patterns: Dict[str, List[Any]] = {}
        self._distractor_patterns: Dict[str, List[Any]] = {}

        if _RDKIT_AVAILABLE and Chem is not None:
            for cat, smarts_list in BOTANICAL_SMARTS_DEFINITIONS.items():
                compiled = []
                for s in smarts_list:
                    try:
                        pat = Chem.MolFromSmarts(s)
                        if pat is not None:
                            compiled.append(pat)
                    except Exception:
                        pass
                self._botanical_patterns[cat] = compiled

            for cat, smarts_list in SYNTHETIC_DISTRACTOR_SMARTS.items():
                compiled = []
                for s in smarts_list:
                    try:
                        pat = Chem.MolFromSmarts(s)
                        if pat is not None:
                            compiled.append(pat)
                    except Exception:
                        pass
                self._distractor_patterns[cat] = compiled

    def _has_match_smiles(self, smi: str, category: str) -> bool:
        """Pure-Python SMILES regex/pattern matcher for environments lacking RDKit."""
        if not smi or not isinstance(smi, str):
            return False

        if category == "sulfonamide":
            # Sulfur double-bonded to two oxygens and bonded to nitrogen
            return bool(
                re.search(r"S\(=O\)\(=O\)[Nn]|S\(=O\)\(=O\)\[n|N[1-9]?\([^)]*\)S\(=O\)\(=O\)|[Nn]S\(=O\)\(=O\)|\[[Nn][^\]]*\]S\(=O\)\(=O\)", smi)
                or "S(=O)(=O)N" in smi
                or "S(=O)(=O)n" in smi
                or "NS(=O)(=O)" in smi
                or "nS(=O)(=O)" in smi
            )

        if category == "halogenated_ring":
            # Aromatic carbon substituted with halogen
            return bool(
                re.search(r"c\d*(?:\([^\)]*\))*(?:Cl|Br|F|I)|(?:Cl|Br|F|I)c\d*", smi)
                or any(x in smi for x in ("c(Cl)", "c(Br)", "c(F)", "c(I)", "(Cl)c", "(Br)c", "(F)c", "(I)c"))
            )

        if category == "hexose":
            # Hexose: sugar with pyranose ring and hydroxymethyl (CO)
            has_sugar_ring = any(p in smi for p in ("OC(", "O)C(", "C(O)C", "C(CO)", "(CO)"))
            return has_sugar_ring and ("(CO)" in smi or "CO)" in smi) and smi.count("O") >= 5

        if category == "rhamnose":
            # 6-deoxyhexose: methyl attached to pyranose ring
            return (
                ("CC1OC" in smi or "C(C)OC" in smi or "CC2OC" in smi or ("CC" in smi and "OC(" in smi))
                and smi.count("O") >= 4
                and "(CO)" not in smi
            )

        if category == "pentose":
            # 5-carbon sugar
            return (
                ("C1OCC" in smi or "OCC1OC" in smi or "OCC" in smi)
                and (3 <= smi.count("O") <= 6)
                and "(CO)" not in smi
            )

        if category == "glucuronide":
            # Glucuronic acid: carboxylic acid on sugar ring
            return (
                ("C(=O)O)C" in smi or "O=C(O)C" in smi or ("C(=O)O" in smi and "OC(" in smi))
                and smi.count("O") >= 6
                and "S(" not in smi
            )

        if category == "galloyl":
            # 3,4,5-trihydroxybenzoyl group (pyrogallol)
            return bool(
                re.search(r"c\(O\)c\(O\)c\(O\)|c\(O\)cc\(O\)c\(O\)|c\(O\)c\(O\)cc\(O\)|\(O\)c\d+cc\(O\)c\(O\)", smi)
                or "c(O)c(O)c(O)" in smi
            )

        if category == "caffeoyl":
            # (E)-3-(3,4-dihydroxyphenyl)acryloyl
            has_caffeic_alkene = any(p in smi for p in ("C=CC(=O)", "/C=C/c", "C=Cc1", "/C=C/C(=O)"))
            has_catechol = "c(O)c(O)" in smi or "c1cc(O)c(O)" in smi or "c(O)cc(O)" in smi
            return has_caffeic_alkene and has_catechol

        if category == "depsidone":
            # Dibenzo-dioxepinone or lactone/depside aryl ester
            return any(p in smi for p in ("C(=O)Oc1", "C(=O)Oc2", "C(=O)Oc3", "c1ccc2c(c1)OC(=O)")) and "O=c1c" not in smi

        if category == "flavonoid":
            # Chromone, flavone, flavonol, flavanone core
            return any(p in smi for p in ("O=c1ccoc2", "O=c1c(O)coc2", "O=C1CCc2c(O1)", "c1ccc2c(=O)ccoc2c1", "O=c1c", "O=C1C("))

        return False

    def _has_match(self, mol: Any, category: str, smiles_str: Optional[str] = None) -> bool:
        """Check if molecule matches category via RDKit SMARTS or SMILES fallback."""
        if _RDKIT_AVAILABLE and Chem is not None and isinstance(mol, Chem.Mol):
            if category in self._botanical_patterns:
                patterns = self._botanical_patterns[category]
            elif category in self._distractor_patterns:
                patterns = self._distractor_patterns[category]
            else:
                patterns = []

            for pat in patterns:
                try:
                    if mol.HasSubstructMatch(pat):
                        return True
                except Exception:
                    continue

        # Extract SMILES string for fallback
        smi = smiles_str
        if not smi and isinstance(mol, str):
            smi = mol
        elif not smi and _RDKIT_AVAILABLE and Chem is not None and isinstance(mol, Chem.Mol):
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = None

        if not smi or not isinstance(smi, str):
            return False

        return self._has_match_smiles(smi, category)

    def detect_spectral_signatures(
        self,
        fused_peaks: Any,
        neutral_mass: float,
        precursor_mz: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        Detect diagnostic botanical neutral losses and fragment ions from MS/MS peaks.

        Args:
            fused_peaks: Array or list of (m/z, intensity) or m/z floats.
            neutral_mass: Precursor monoisotopic neutral mass in Daltons.
            precursor_mz: Optional observed precursor m/z.

        Returns:
            Dict mapping signature names to boolean presence.
        """
        signatures: Dict[str, bool] = {
            "hexose": False,
            "rhamnose": False,
            "pentose": False,
            "glucuronide": False,
            "galloyl": False,
            "caffeoyl": False,
            "depsidone": False,
            "flavonoid": False,
        }

        peak_mzs = extract_peak_mzs(fused_peaks)
        if not peak_mzs:
            return signatures

        # Calculate candidate neutral losses for each peak
        candidate_nls: List[float] = []
        if neutral_mass > 0.0:
            for mz in peak_mzs:
                if mz < neutral_mass + 5.0:
                    candidate_nls.append(neutral_mass - mz)
                    candidate_nls.append((neutral_mass + 1.007825) - mz)
                    candidate_nls.append((neutral_mass - 1.007825) - mz)
        if precursor_mz is not None and precursor_mz > 0.0:
            for mz in peak_mzs:
                if mz < precursor_mz + 5.0:
                    candidate_nls.append(precursor_mz - mz)

        def matches_any_nl(target: float, tol: Optional[float] = None) -> bool:
            t = tol or self.tolerance_da
            return any(abs(nl - target) <= t for nl in candidate_nls)

        def matches_any_peak(target: float, tol: Optional[float] = None) -> bool:
            t = tol or self.tolerance_da
            return any(abs(mz - target) <= t for mz in peak_mzs)

        # 1. Hexose loss: 162.053 +/- 0.025 Da
        # 6. Caffeoyl loss: 162.032 +/- 0.025 Da
        nl_near_162 = [nl for nl in candidate_nls if abs(nl - 162.042) <= 0.040]
        if nl_near_162:
            best_nl = min(nl_near_162, key=lambda x: min(abs(x - 162.053), abs(x - 162.032)))
            d_hex = abs(best_nl - 162.053)
            d_caff = abs(best_nl - 162.032)
            if d_caff < d_hex and d_caff <= self.tolerance_da:
                signatures["caffeoyl"] = True
            elif d_hex <= self.tolerance_da:
                signatures["hexose"] = True
        elif matches_any_nl(162.053):
            signatures["hexose"] = True

        # 2. Rhamnose / Deoxyhexose loss: 146.058 +/- 0.025 Da
        if matches_any_nl(146.058):
            signatures["rhamnose"] = True

        # 3. Pentose loss: 132.042 +/- 0.025 Da
        if matches_any_nl(132.042):
            signatures["pentose"] = True

        # 4. Glucuronide loss: 176.032 +/- 0.025 Da
        if matches_any_nl(176.032):
            signatures["glucuronide"] = True

        # 5. Galloyl loss: 152.011 +/- 0.025 Da or peak at 153.019 +/- 0.025 Da
        if matches_any_nl(152.011) or matches_any_peak(153.019):
            signatures["galloyl"] = True

        # 7. Depsidone / lactone ester: diagnostic neutral loss or lactone fragment
        # CO2 loss (43.990), CO loss (27.995), or depsidone fragment (168.040)
        if (
            matches_any_nl(43.990)
            or matches_any_nl(27.995)
            or matches_any_nl(168.040)
            or matches_any_peak(168.040)
        ):
            signatures["depsidone"] = True

        # 8. Flavonoid / chromone core: retro-Diels-Alder fragments (153.019, 137.024, 121.029)
        if (
            matches_any_peak(153.019)
            or matches_any_peak(137.024)
            or matches_any_peak(121.029)
        ):
            signatures["flavonoid"] = True

        return signatures

    def score_candidate(
        self,
        mol: Any,
        fused_peaks: Optional[Any],
        neutral_mass: float,
        precursor_mz: Optional[float] = None,
    ) -> float:
        """
        Evaluate candidate Mol against detected botanical MS/MS signatures.

        Args:
            mol: RDKit Mol or SMILES string.
            fused_peaks: Query MS/MS spectrum peaks.
            neutral_mass: Precursor neutral mass in Daltons.
            precursor_mz: Optional observed precursor m/z.

        Returns:
            botanical_score in range [0.0, 1.0].
        """
        if mol is None:
            return 0.0

        m = None
        smi = None
        if isinstance(mol, str):
            smi = mol.strip()
            if _RDKIT_AVAILABLE and Chem is not None:
                try:
                    m = Chem.MolFromSmiles(smi)
                except Exception:
                    m = None
                if m is None:
                    return 0.0
            else:
                # Basic chemical sanity check when RDKit is absent
                if any(c in smi for c in "_jJqQxXzZ") or not any(c in smi for c in "CcNnOoSs"):
                    return 0.0
        elif _RDKIT_AVAILABLE and Chem is not None and isinstance(mol, Chem.Mol):
            m = mol
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = None

        if m is None and not smi:
            return 0.0

        # Substructure matching for candidate
        cand_has: Dict[str, bool] = {
            cat: self._has_match(m, cat, smiles_str=smi)
            for cat in [
                "hexose",
                "rhamnose",
                "pentose",
                "glucuronide",
                "galloyl",
                "caffeoyl",
                "depsidone",
                "flavonoid",
                "sulfonamide",
                "halogenated_ring",
            ]
        }

        # Detect signatures in observed spectrum
        spec_sig = self.detect_spectral_signatures(fused_peaks, neutral_mass, precursor_mz)
        has_botanical = any(spec_sig.values())

        is_distractor = cand_has["sulfonamide"] or cand_has["halogenated_ring"]

        if has_botanical:
            # Spectrum exhibits pure natural product fragmentation patterns
            if is_distractor:
                # Severe penalty for synthetic sulfonamides or halogenated rings
                penalty = 0.05
                if cand_has["sulfonamide"] and cand_has["halogenated_ring"]:
                    penalty = 0.01
                return penalty

            # Non-distractor candidate: score botanical alignment
            score = 0.60
            matched_count = 0
            mismatched_count = 0

            for cat in [
                "hexose",
                "rhamnose",
                "pentose",
                "glucuronide",
                "galloyl",
                "caffeoyl",
                "depsidone",
                "flavonoid",
            ]:
                if spec_sig.get(cat, False):
                    if cand_has.get(cat, False):
                        matched_count += 1
                        score += 0.30
                    else:
                        mismatched_count += 1
                        score -= 0.30

            # Neutral loss check: If sugar loss (hexose, rhamnose, pentose, glucuronide) is detected,
            # the candidate MUST possess that sugar moiety; otherwise it is an aglycone or mismatch
            has_sugar_loss = any(
                spec_sig.get(s, False)
                for s in ["hexose", "rhamnose", "pentose", "glucuronide"]
            )
            has_matching_sugar = any(
                spec_sig.get(s, False) and cand_has.get(s, False)
                for s in ["hexose", "rhamnose", "pentose", "glucuronide"]
            )
            if has_sugar_loss and not has_matching_sugar:
                score = min(0.50, score)
            elif matched_count > 0:
                score = min(0.99, max(0.85, score))
            elif mismatched_count > 0:
                score = max(0.15, min(0.45, score))

            return float(np.clip(score, 0.01, 0.99))

        else:
            # No strong botanical signatures detected in spectrum
            if cand_has["sulfonamide"]:
                return 0.15
            if cand_has["halogenated_ring"]:
                return 0.25
            has_any_np = any(
                cand_has[cat]
                for cat in [
                    "hexose",
                    "rhamnose",
                    "pentose",
                    "glucuronide",
                    "galloyl",
                    "caffeoyl",
                    "depsidone",
                    "flavonoid",
                ]
            )
            if has_any_np:
                return 0.60
            return 0.50
