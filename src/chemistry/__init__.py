"""
Chemical foundation module for CASMI 2026.
Contains adduct mass calculation, 13C mispick correction, and RDKit molecular standardization.
"""

from src.chemistry.adducts import (
    COMPETITION_ADDUCTS,
    CARBON_13_DELTA,
    ELECTRON_MASS,
    PROTON_MASS,
    AdductInfo,
    calculate_neutral_mass,
    calculate_precursor_mz,
    get_adduct_candidates,
    get_adduct_info,
    normalize_adduct_name,
    detect_ms2_precursor_cluster,
    correct_precursor_mz,
    get_precursor_hypotheses,
)
from src.chemistry.standardizer import (
    standardize_mol,
    get_inchikey14,
    deduplicate_candidates,
    deduplicate_smiles_list,
)

__all__ = [
    "COMPETITION_ADDUCTS",
    "CARBON_13_DELTA",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "AdductInfo",
    "calculate_neutral_mass",
    "calculate_precursor_mz",
    "get_adduct_candidates",
    "get_adduct_info",
    "normalize_adduct_name",
    "detect_ms2_precursor_cluster",
    "correct_precursor_mz",
    "get_precursor_hypotheses",
    "standardize_mol",
    "get_inchikey14",
    "deduplicate_candidates",
    "deduplicate_smiles_list",
]
