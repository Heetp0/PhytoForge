"""
Adduct definitions, monoisotopic delta mass calculations, and precursor correction
for the Enveda CASMI 2026 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Physical Constants (NIST / IUPAC Monoisotopic Weights in Daltons)
ELECTRON_MASS: float = 0.000548579909
CARBON_13_DELTA: float = 1.003355  # 13C - 12C delta mass in Daltons
PROTON_MASS: float = 1.007276452321     # 1H (1.00782503223) - electron_mass


@dataclass(frozen=True)
class AdductInfo:
    """Metadata and exact mass delta for an electrospray adduct."""
    name: str
    polarity: str       # 'positive' or 'negative'
    charge: int         # Signed integer: +1, -1, +2, -2
    mult: int           # Multiplicity n (1 for monomer, 2 for dimer)
    delta_mass: float   # Monoisotopic delta mass: m_ion - n*M (in Daltons)

    @property
    def abs_charge(self) -> int:
        return abs(self.charge)


# The 10 Official Competition Adducts (calculated to 9 decimal places)
COMPETITION_ADDUCTS: Dict[str, AdductInfo] = {
    # Positive Mode (Cations, z = +1, n = 1)
    "[M+H]+": AdductInfo(
        name="[M+H]+",
        polarity="positive",
        charge=1,
        mult=1,
        delta_mass=1.007276452,
    ),
    "[M+NH4]+": AdductInfo(
        name="[M+NH4]+",
        polarity="positive",
        charge=1,
        mult=1,
        delta_mass=18.033825553,
    ),
    "[M+Na]+": AdductInfo(
        name="[M+Na]+",
        polarity="positive",
        charge=1,
        mult=1,
        delta_mass=22.989220702,
    ),
    "[M+K]+": AdductInfo(
        name="[M+K]+",
        polarity="positive",
        charge=1,
        mult=1,
        delta_mass=38.963157906,
    ),
    "[M-H2O+H]+": AdductInfo(
        name="[M-H2O+H]+",
        polarity="positive",
        charge=1,
        mult=1,
        delta_mass=-17.003288232,
    ),
    # Negative Mode (Anions, z = -1, n = 1)
    "[M-H]-": AdductInfo(
        name="[M-H]-",
        polarity="negative",
        charge=-1,
        mult=1,
        delta_mass=-1.007276452,
    ),
    "[M+Cl]-": AdductInfo(
        name="[M+Cl]-",
        polarity="negative",
        charge=-1,
        mult=1,
        delta_mass=34.969401301,
    ),
    "[M+FA-H]-": AdductInfo(
        name="[M+FA-H]-",
        polarity="negative",
        charge=-1,
        mult=1,
        delta_mass=44.998202851,
    ),
    "[M+Hac-H]-": AdductInfo(
        name="[M+Hac-H]-",
        polarity="negative",
        charge=-1,
        mult=1,
        delta_mass=59.013852916,
    ),
    "[M-H2O-H]-": AdductInfo(
        name="[M-H2O-H]-",
        polarity="negative",
        charge=-1,
        mult=1,
        delta_mass=-19.017841136,
    ),
}

# Extended Adducts (for physical plausibility & multi-water-loss modeling)
EXTENDED_ADDUCTS: Dict[str, AdductInfo] = {
    "[M-2H2O+H]+": AdductInfo(
        name="[M-2H2O+H]+",
        polarity="positive",
        charge=1,
        mult=1,
        delta_mass=-35.013852916,
    ),
    "[M-2H2O-H]-": AdductInfo(
        name="[M-2H2O-H]-",
        polarity="negative",
        charge=-1,
        mult=1,
        delta_mass=-37.028405820,
    ),
}

# Common spelling / formatting aliases mapped to canonical names
ADDUCT_ALIASES: Dict[str, str] = {
    "[M+HCOO]-": "[M+FA-H]-",
    "[M+CH3COO]-": "[M+Hac-H]-",
    "[M+OAC]-": "[M+Hac-H]-",
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M-H-H2O]-": "[M-H2O-H]-",
    "[M+H-2H2O]+": "[M-2H2O+H]+",
    "[M-H-2H2O]-": "[M-2H2O-H]-",
}


def normalize_adduct_name(adduct: str) -> str:
    """Sanitizes and normalizes adduct string formatting."""
    cleaned = adduct.strip().replace(" ", "")
    upper = cleaned.upper()
    # Check alias matches
    if upper in ADDUCT_ALIASES:
        return ADDUCT_ALIASES[upper]
    for canonical in COMPETITION_ADDUCTS:
        if canonical.upper() == upper:
            return canonical
    for canonical in EXTENDED_ADDUCTS:
        if canonical.upper() == upper:
            return canonical
    return cleaned


def get_adduct_info(adduct: str) -> AdductInfo:
    """
    Retrieves AdductInfo for a given adduct string.
    Raises ValueError if the adduct is not recognized.
    """
    norm = normalize_adduct_name(adduct)
    if norm in COMPETITION_ADDUCTS:
        return COMPETITION_ADDUCTS[norm]
    if norm in EXTENDED_ADDUCTS:
        return EXTENDED_ADDUCTS[norm]
    raise ValueError(
        f"Unknown or unsupported adduct: '{adduct}'. "
        f"Supported adducts: {list(COMPETITION_ADDUCTS.keys())}"
    )


def calculate_neutral_mass(precursor_mz: float, adduct: str) -> float:
    """
    Calculates the neutral monoisotopic mass M from precursor m/z and adduct.
    
    Formula: M = (|z| * (m/z) - delta_mass) / mult
    """
    if precursor_mz <= 0.0:
        raise ValueError(f"Precursor m/z must be positive, got: {precursor_mz}")
    info = get_adduct_info(adduct)
    return (info.abs_charge * float(precursor_mz) - info.delta_mass) / info.mult


def calculate_precursor_mz(neutral_mass: float, adduct: str) -> float:
    """
    Calculates expected precursor m/z from neutral monoisotopic mass and adduct.
    
    Formula: m/z = (mult * M + delta_mass) / |z|
    """
    if neutral_mass <= 0.0:
        raise ValueError(f"Neutral mass must be positive, got: {neutral_mass}")
    info = get_adduct_info(adduct)
    return (info.mult * float(neutral_mass) + info.delta_mass) / info.abs_charge


def get_adduct_candidates(precursor_mz: float, mode: str) -> List[Tuple[str, float]]:
    """
    Returns candidate (adduct_name, neutral_mass) pairs for all supported adducts
    matching the specified ionization polarity ('positive' or 'negative').
    Filters out physically impossible non-positive neutral masses.
    """
    norm_mode = mode.strip().lower()
    if norm_mode in ("pos", "positive", "+", "1"):
        target_pol = "positive"
    elif norm_mode in ("neg", "negative", "-", "-1"):
        target_pol = "negative"
    else:
        raise ValueError(f"Invalid ionization mode: '{mode}'. Expected 'positive' or 'negative'.")

    candidates: List[Tuple[str, float]] = []
    for name, info in COMPETITION_ADDUCTS.items():
        if info.polarity == target_pol:
            m = calculate_neutral_mass(precursor_mz, name)
            if m > 0.0:
                candidates.append((name, m))
    return candidates


def detect_ms2_precursor_cluster(
    observed_mz: float,
    ms2_peaks: Optional[np.ndarray],
    tolerance_da: float = 0.02,
    min_intensity_ratio: float = 0.05,
) -> bool:
    """
    Scans MS2 spectrum for residual unfragmented monoisotopic precursor peak
    at (observed_mz - 1.003355 Da), which indicates that the instrument peak picker
    mispicked the [M+1] isotopic peak in the quadrupole window.
    
    Args:
        observed_mz: The precursor m/z recorded in metadata.
        ms2_peaks: (N, 2) array of [mz, intensity].
        tolerance_da: Absolute mass matching window in Daltons.
        min_intensity_ratio: Minimum fraction of base peak intensity.
        
    Returns:
        True if an MS2 peak matching (observed_mz - 1.003355) is present.
    """
    if ms2_peaks is None or len(ms2_peaks) == 0:
        return False

    if not isinstance(ms2_peaks, np.ndarray) or ms2_peaks.ndim != 2 or ms2_peaks.shape[1] < 2:
        return False

    mzs = ms2_peaks[:, 0]
    intensities = ms2_peaks[:, 1]
    max_intensity = float(np.max(intensities)) if len(intensities) > 0 else 1.0
    if max_intensity <= 0.0:
        return False

    target_m0_mz = observed_mz - CARBON_13_DELTA
    diffs = np.abs(mzs - target_m0_mz)
    matching_indices = np.where(diffs <= tolerance_da)[0]

    for idx in matching_indices:
        if (intensities[idx] / max_intensity) >= min_intensity_ratio:
            return True

    return False


def correct_precursor_mz(
    observed_mz: float,
    ms2_peaks: Optional[np.ndarray] = None,
    tolerance_da: float = 0.02,
) -> float:
    """
    Corrects precursor m/z for false +1 13C quadrupole mispicks if spectral
    evidence confirms residual monoisotopic ion in MS2. Falls back to observed_mz.
    """
    if ms2_peaks is not None and len(ms2_peaks) > 0:
        if detect_ms2_precursor_cluster(observed_mz, ms2_peaks, tolerance_da=tolerance_da):
            return observed_mz - CARBON_13_DELTA
    return observed_mz


def get_precursor_hypotheses(
    observed_mz: float,
    ms2_peaks: Optional[np.ndarray] = None,
    tolerance_da: float = 0.02,
) -> List[Tuple[float, float, str]]:
    """
    Generates ranked precursor hypotheses: (mz, confidence_weight, hypothesis_label).
    
    If MS2 spectral confirmation detects the M+0 peak, the 13C mispick hypothesis
    is elevated to primary rank (weight 1.0). Otherwise, nominal observed_mz is rank 1.
    """
    is_confirmed_mispick = False
    if ms2_peaks is not None and len(ms2_peaks) > 0:
        is_confirmed_mispick = detect_ms2_precursor_cluster(
            observed_mz, ms2_peaks, tolerance_da=tolerance_da
        )

    if is_confirmed_mispick:
        return [
            (observed_mz - CARBON_13_DELTA, 1.0, "13C_confirmed_mispick"),
            (observed_mz, 0.50, "nominal_fallback"),
        ]

    return [
        (observed_mz, 1.0, "nominal"),
        (observed_mz - CARBON_13_DELTA, 0.85, "13C_mispick_fallback"),
        (observed_mz - 2.0 * CARBON_13_DELTA, 0.40, "13C2_mispick_fallback"),
    ]


def parse_formula_oxygens(formula: str) -> int:
    """Extract count of oxygen atoms from a molecular formula string."""
    match = re.search(r"O(?![a-z])(\d*)", formula)
    if not match:
        return 0
    qty = match.group(1)
    return int(qty) if qty else 1


def calculate_canonical_neutral_mass(
    mz: float,
    adduct: str,
    formula_oxygens: Optional[int] = None,
    formula: Optional[str] = None,
) -> float:
    """
    Calculate canonical neutral monoisotopic mass with physical plausibility validation.
    
    Enforces that:
    - Double water loss ([M-2H2O+H]+, [M-2H2O-H]-) requires at least 2 oxygen atoms.
    - Single water loss ([M-H2O+H]+, [M-H2O-H]-) requires at least 1 oxygen atom.
    
    Args:
        mz: Precursor m/z value.
        adduct: Adduct string identifier.
        formula_oxygens: Explicit oxygen atom count if known.
        formula: Molecular formula string (e.g. 'C15H10O7') from which oxygen count is parsed.
        
    Returns:
        Canonical neutral monoisotopic mass in Daltons.
        
    Raises:
        ValueError: If physical feasibility constraints are violated or adduct is invalid.
    """
    norm_adduct = normalize_adduct_name(adduct)

    if formula_oxygens is None and formula is not None:
        formula_oxygens = parse_formula_oxygens(formula)

    if formula_oxygens is not None:
        if "-2H2O" in norm_adduct and formula_oxygens < 2:
            raise ValueError(f"{norm_adduct} requires at least 2 oxygen atoms in the molecule")
        if "-H2O" in norm_adduct and "-2H2O" not in norm_adduct and formula_oxygens < 1:
            raise ValueError(f"{norm_adduct} requires at least 1 oxygen atom in the molecule")

    return calculate_neutral_mass(mz, norm_adduct)


def get_multihypothesis_precursor_candidates(
    mz: float,
    adduct: str,
    mw_estimate: Optional[float] = None,
) -> List[Tuple[float, str, float]]:
    """
    Return neutral mass hypotheses including M0, M-1 (13C), and M-2 (double 13C for MW >= 400).
    
    Args:
        mz: Precursor m/z value.
        adduct: Adduct string.
        mw_estimate: Optional molecular weight estimate. If None, derived from nominal mass.
        
    Returns:
        List of (neutral_mass, hypothesis_label, prior_weight) tuples.
    """
    norm_adduct = normalize_adduct_name(adduct)
    base_m = calculate_neutral_mass(mz, norm_adduct)
    results = [(base_m, "M0", 1.0)]

    # 13C offset 1 (single 13C mispick)
    results.append((base_m - CARBON_13_DELTA, "M-1_13C", 0.35))

    # 13C offset 2 for large molecules (double 13C mispick)
    effective_mw = mw_estimate if mw_estimate is not None else base_m
    if effective_mw >= 400.0:
        results.append((base_m - 2.0 * CARBON_13_DELTA, "M-2_13C", 0.08))

    return results
