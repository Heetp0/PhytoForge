"""Tests for [M+1] isotope envelope carbon estimation and ExactMassFormulaIndex."""

import numpy as np
import pytest
from src.chemistry.adducts import CARBON_13_DELTA, estimate_carbon_count_from_isotope_cluster
from src.retrieval.mist_formula_router import ExactMassFormulaIndex


def test_estimate_carbon_count_ideal_cluster():
    """Verify carbon estimation on synthetic Quercetin isotope cluster (C15H10O7, MW 302.0427)."""
    p_mz = 303.0500  # [M+H]+
    true_carbons = 15

    # Natural abundance: 15 * 0.0108 = 0.162 of M0
    peaks = np.array([
        [303.0500, 1.000],                       # M0
        [303.0500 + CARBON_13_DELTA, 0.162],    # M+1
        [153.0180, 0.500],                       # Fragment
    ], dtype=np.float64)

    est_c = estimate_carbon_count_from_isotope_cluster(p_mz, peaks, tolerance_da=0.02)
    assert est_c is not None
    assert round(est_c) == true_carbons


def test_estimate_carbon_count_no_m1():
    """Returns None when no M+1 peak is present."""
    p_mz = 303.0500
    peaks = np.array([
        [303.0500, 1.000],
        [153.0180, 0.500],
    ], dtype=np.float64)

    est_c = estimate_carbon_count_from_isotope_cluster(p_mz, peaks, tolerance_da=0.02)
    assert est_c is None


def test_estimate_carbon_count_empty_or_low_intensity():
    """Returns None on empty peaks or low precursor intensity."""
    assert estimate_carbon_count_from_isotope_cluster(303.05, None) is None
    assert estimate_carbon_count_from_isotope_cluster(303.05, np.zeros((0, 2))) is None

    # M0 too weak (< 0.01)
    peaks = np.array([[303.05, 0.005], [304.053, 0.001]])
    assert estimate_carbon_count_from_isotope_cluster(303.05, peaks) is None


def test_exact_mass_formula_index_basic():
    """Test ExactMassFormulaIndex formula mass calculation and tolerance lookup."""
    formulas = [
        "C15H10O7",  # Quercetin: MW 302.0427
        "C15H10O5",  # Apigenin: MW 270.0528
        "C6H6O",     # Phenol: MW 94.0418
    ]
    idx = ExactMassFormulaIndex(formulas=formulas)
    assert len(idx.formulas) == 3

    # Exact lookup for Quercetin
    matches = idx.find_formulas_by_mass(target_mass=302.0427, tolerance_ppm=15.0)
    assert len(matches) == 1
    assert matches[0][0] == "C15H10O7"
    assert matches[0][1] > 0.90


def test_exact_mass_formula_index_carbon_prioritization():
    """Verify that carbon count estimate prioritizes the formula matching the estimated carbon skeleton."""
    formulas = [
        "C15H10O7",   # MW ~ 302.0427, 15 carbons
        "C14H6O8",    # MW ~ 302.0063, 14 carbons
        "C16H14O6",   # MW ~ 302.0790, 16 carbons
    ]
    idx = ExactMassFormulaIndex(formulas=formulas)

    # Wide 150 ppm tolerance so all 3 match
    matches_no_c = idx.find_formulas_by_mass(target_mass=302.04, tolerance_ppm=150.0)
    assert len(matches_no_c) == 3

    # With carbon estimate = 15.0, C15H10O7 should be top-ranked
    matches_c15 = idx.find_formulas_by_mass(
        target_mass=302.04,
        tolerance_ppm=150.0,
        target_carbon_estimate=15.0,
    )
    assert matches_c15[0][0] == "C15H10O7"
