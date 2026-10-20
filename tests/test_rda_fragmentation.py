import pytest
import numpy as np
from rdkit import Chem
from src.pipeline import _compute_candidate_fragment_match_ratio


def test_quercetin_rda_fragment_matching():
    """Verify that Quercetin generates authentic RDA 1,3A+ (m/z ~153.0) and 0,2A+ (m/z ~137.0) ions."""
    # Quercetin SMILES (MW 302.0427)
    smi = "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None

    # Observed MS/MS spectrum containing the true Quercetin RDA fragments + precursor
    peaks = np.array([
        [303.0500, 100.0],  # [M+H]+ precursor
        [153.0182, 85.0],   # 1,3A+ RDA ion (base diagnostic peak)
        [137.0233, 45.0],   # 0,2A+ RDA ion
        [285.0400, 30.0],   # [M+H-H2O]+
        [179.0300, 20.0],   # 1,2A+ RDA ion
    ], dtype=np.float64)

    ratio = _compute_candidate_fragment_match_ratio(mol, peaks, 302.0427)
    # The true Quercetin candidate should match the RDA peaks with high ratio
    assert ratio >= 0.5, f"Expected ratio >= 0.5 for Quercetin with authentic RDA peaks, got {ratio}"


def test_intensity_weighted_fragment_matching():
    """Verify that matching high-intensity base peaks yields a significantly higher ratio than baseline noise."""
    smi = "O=C(Oc1ccccc1)c1ccccc1"  # Phenyl benzoate (MW 198.0681)
    mol = Chem.MolFromSmiles(smi)

    # Spectrum 1: matches the base peak at 105.034 (100% intensity)
    peaks_major = np.array([
        [105.034, 100.0],  # Major fragment matching benzoyl ion
        [77.039, 80.0],    # Major fragment matching phenyl ion
        [50.0, 5.0],       # Unmatched noise
    ], dtype=np.float64)

    # Spectrum 2: matches only a tiny 2% noise peak at 105.034
    peaks_minor = np.array([
        [105.034, 2.0],    # Minor match
        [150.0, 100.0],    # Unmatched base peak
        [180.0, 90.0],     # Unmatched high peak
    ], dtype=np.float64)

    ratio_major = _compute_candidate_fragment_match_ratio(mol, peaks_major, 198.0681)
    ratio_minor = _compute_candidate_fragment_match_ratio(mol, peaks_minor, 198.0681)

    assert ratio_major > ratio_minor + 0.15, (
        f"Major peak match ({ratio_major}) should exceed minor match ({ratio_minor}) by >= 0.15"
    )
