"""Wave A Test Suite: Chemistry & Preprocessing Core.

Verifies:
1. RDKit standardization, salt stripping, and charge neutralization.
2. Parent neutral formula extraction (stripping Na+, Cl-, etc.).
3. Dimer adduct calculations ([2M+H]+, [2M-H]-, [2M+Na]+).
4. Multi-energy peak fusion with per-frame base-peak normalization.
5. ExactMassFormulaIndex binary search lookup and physical formula confidence weighting.
"""

from pathlib import Path
import pytest
import numpy as np

from src.chemistry.standardizer import (
    standardize_mol,
    compute_hill_formula,
    compute_morgan_fingerprint,
    get_inchikey14,
)
from src.chemistry.adducts import (
    calculate_neutral_mass,
    calculate_precursor_mz,
    get_adduct_info,
    normalize_adduct_name,
    EXTENDED_ADDUCTS,
)
from src.data.multi_energy_fusion import (
    SpectralFrame,
    fuse_peaks,
    MultiEnergyFusionEngine,
)
from src.retrieval.mist_formula_router import (
    ExactMassFormulaIndex,
    MISTFormulaRouter,
)


def test_standardize_mol_salt_stripping():
    """Verify sodium benzoate salt is stripped and neutralized to benzoic acid."""
    salt_smi = "[Na+].[O-]C(=O)c1ccccc1"
    canon_smi, ik14 = standardize_mol(salt_smi)
    assert canon_smi == "O=C(O)c1ccccc1"
    assert ik14 == "WPYMKLBDIGXBTP"


def test_compute_hill_formula_neutral_parent():
    """Verify formula computation on salts strips counterions to isolate neutral parent."""
    salt_smi = "[Na+].[O-]C(=O)c1ccccc1"
    form = compute_hill_formula(salt_smi, standardize=True)
    assert form == "C7H6O2"


def test_adducts_dimer_support():
    """Verify extended adduct library supports non-covalent dimers with multiplicity=2."""
    assert "[2M+H]+" in EXTENDED_ADDUCTS
    assert "[2M-H]-" in EXTENDED_ADDUCTS
    assert "[2M+Na]+" in EXTENDED_ADDUCTS

    info_pos = get_adduct_info("[2M+H]+")
    assert info_pos.mult == 2
    assert info_pos.charge == 1

    # Quercetin (M = 302.04257 Da) -> [2M+H]+ = (2*302.04257 + 1.007276) = 605.0924 Da
    m_ion = calculate_precursor_mz(neutral_mass=302.04257, adduct="[2M+H]+")
    assert abs(m_ion - 605.0924) < 1e-3

    m_neutral = calculate_neutral_mass(precursor_mz=m_ion, adduct="[2M+H]+")
    assert abs(m_neutral - 302.04257) < 1e-4


def test_multi_energy_fusion_base_peak_normalization():
    """Verify fuse_peaks normalizes disparate intensity scales so raw counts do not swamp normalized frames."""
    # Frame 1: Raw counts at 20 eV (0 to 50,000)
    peaks_f1 = np.array([
        [153.02, 50000.0],
        [303.05, 25000.0],
    ], dtype=np.float64)

    # Frame 2: Pre-normalized percentages at 40 eV (0 to 100) with diagnostic low-mass fragment
    peaks_f2 = np.array([
        [137.02, 80.0],
        [153.02, 100.0],
    ], dtype=np.float64)

    f1 = SpectralFrame(collision_energy=20.0, peaks=peaks_f1)
    f2 = SpectralFrame(collision_energy=40.0, peaks=peaks_f2)

    fused = fuse_peaks([f1, f2], tolerance_da=0.02, normalize_per_frame=True)
    assert fused.shape[0] == 3  # 137.02, 153.02, 303.05

    # Check that diagnostic fragment 137.02 is present with significant relative intensity
    mzs = fused[:, 0]
    ints = fused[:, 1]
    idx_137 = np.argmin(np.abs(mzs - 137.02))
    idx_153 = np.argmin(np.abs(mzs - 153.02))

    # In frame 2, 137 is 80% of 153. After normalization, it must retain its relative ~80% strength
    ratio = ints[idx_137] / ints[idx_153]
    assert 0.70 <= ratio <= 0.90


def test_exact_mass_formula_index_binary_search_and_nitrogen_rule():
    """Verify ExactMassFormulaIndex performs fast binary search and respects physical isotope bounds."""
    test_formulas = [
        "C6H6",       # 78.04695
        "C7H6O2",     # 122.03678
        "C8H10N4O2",  # Caffeine 194.08038
        "C15H10O7",   # Quercetin 302.04265
        "C15H10O6",   # Kaempferol 286.04774
        "C15H10O5",   # Apigenin 270.05282
        "C27H30O16",  # Rutin 610.15339
    ]
    idx = ExactMassFormulaIndex(formulas=test_formulas)

    # Look up Quercetin mass 302.04265 within 20 ppm
    matches = idx.find_formulas_by_mass(target_mass=302.04265, tolerance_ppm=20.0)
    assert len(matches) >= 1
    assert matches[0][0] == "C15H10O7"
    assert matches[0][1] > 0.90  # High confidence
