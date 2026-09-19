import pytest
import numpy as np
from src.chemistry.adducts import (
    COMPETITION_ADDUCTS,
    EXTENDED_ADDUCTS,
    calculate_neutral_mass,
    calculate_precursor_mz,
    calculate_canonical_neutral_mass,
    get_multihypothesis_precursor_candidates,
    parse_formula_oxygens,
    CARBON_13_DELTA,
    normalize_adduct_name,
    get_adduct_candidates,
    detect_ms2_precursor_cluster,
    correct_precursor_mz,
    get_precursor_hypotheses,
    AdductInfo
)

def test_competition_adducts_roundtrip():
    base_mass = 500.1234
    for adduct in COMPETITION_ADDUCTS:
        mz = calculate_precursor_mz(base_mass, adduct)
        mass_calc = calculate_neutral_mass(mz, adduct)
        np.testing.assert_allclose(mass_calc, base_mass, rtol=1e-6)

def test_extended_adducts_roundtrip():
    base_mass = 250.0
    for adduct in EXTENDED_ADDUCTS:
        mz = calculate_precursor_mz(base_mass, adduct)
        mass_calc = calculate_neutral_mass(mz, adduct)
        np.testing.assert_allclose(mass_calc, base_mass, rtol=1e-6)

def test_canonical_neutral_mass_unknown_adduct():
    with pytest.raises(ValueError):
        calculate_canonical_neutral_mass(100.0, "[M+Unknown]+")

def test_canonical_neutral_mass_negative_mz():
    with pytest.raises(ValueError):
        calculate_canonical_neutral_mass(-50.0, "[M+H]+")

def test_get_multihypothesis_precursor_candidates_count():
    res = get_multihypothesis_precursor_candidates(200.0, "[M+H]+")
    assert len(res) == 2  # base_mass < 400, so M0 and M-1_13C only
    
    res2 = get_multihypothesis_precursor_candidates(500.0, "[M+H]+")
    assert len(res2) == 3 # base_mass >= 400, so M0, M-1, M-2

def test_single_water_loss_plausibility():
    with pytest.raises(ValueError, match="requires at least 1 oxygen"):
        calculate_canonical_neutral_mass(100.0, "[M-H2O+H]+", formula_oxygens=0)
    # Should not raise for O>=1
    calculate_canonical_neutral_mass(100.0, "[M-H2O+H]+", formula_oxygens=1)

def test_double_water_loss_plausibility():
    with pytest.raises(ValueError, match="requires at least 2 oxygen"):
        calculate_canonical_neutral_mass(100.0, "[M-2H2O+H]+", formula_oxygens=1)
    # Should not raise for O>=2
    calculate_canonical_neutral_mass(100.0, "[M-2H2O+H]+", formula_oxygens=2)

def test_parse_formula_oxygens_edge_cases():
    assert parse_formula_oxygens("C6H12") == 0
    assert parse_formula_oxygens("CH4O") == 1
    assert parse_formula_oxygens("C10H20O10") == 10
    assert parse_formula_oxygens("OsO4") == 4
    assert parse_formula_oxygens("Os") == 0

def test_carbon_13_delta_precision():
    assert abs(CARBON_13_DELTA - 1.003355) < 1e-6

def test_13c_mispick_correction():
    mz = 200.0
    ms2_peaks = np.array([
        [mz - CARBON_13_DELTA, 1000.0],
        [mz, 100.0]
    ])
    corrected = correct_precursor_mz(mz, ms2_peaks)
    assert corrected == mz - CARBON_13_DELTA

def test_normalize_adduct_name_whitespace_case():
    assert normalize_adduct_name("  [M + h]+ ") == "[M+H]+"
    assert normalize_adduct_name("[M+Hac-H]-") == "[M+Hac-H]-"
    assert normalize_adduct_name("[m+CH3COO]-") == "[M+Hac-H]-"

def test_get_adduct_candidates():
    pos_candidates = get_adduct_candidates(200.0, "positive")
    assert len(pos_candidates) == len([a for a, i in COMPETITION_ADDUCTS.items() if i.polarity == "positive"])
    neg_candidates = get_adduct_candidates(200.0, "-1")
    assert len(neg_candidates) == len([a for a, i in COMPETITION_ADDUCTS.items() if i.polarity == "negative"])

def test_detect_ms2_precursor_cluster_empty():
    assert not detect_ms2_precursor_cluster(200.0, None)
    assert not detect_ms2_precursor_cluster(200.0, np.array([]))

def test_detect_ms2_precursor_cluster_valid():
    mz = 300.0
    peaks = np.array([
        [150.0, 500.0],
        [mz - CARBON_13_DELTA, 100.0]
    ])
    assert detect_ms2_precursor_cluster(mz, peaks, min_intensity_ratio=0.1)
    # If ratio is too high, should fail
    assert not detect_ms2_precursor_cluster(mz, peaks, min_intensity_ratio=0.5)

def test_get_precursor_hypotheses_no_ms2():
    hyps = get_precursor_hypotheses(200.0)
    assert len(hyps) == 3
    assert hyps[0][2] == "nominal"
    assert hyps[1][2] == "13C_mispick_fallback"
    assert hyps[2][2] == "13C2_mispick_fallback"

def test_get_precursor_hypotheses_with_ms2():
    mz = 200.0
    peaks = np.array([[mz - CARBON_13_DELTA, 100.0]])
    hyps = get_precursor_hypotheses(mz, peaks)
    assert len(hyps) == 2
    assert hyps[0][2] == "13C_confirmed_mispick"
    assert hyps[1][2] == "nominal_fallback"

def test_adduct_info_abs_charge():
    a1 = AdductInfo("A", "pos", 2, 1, 1.0)
    a2 = AdductInfo("B", "neg", -3, 1, -1.0)
    assert a1.abs_charge == 2
    assert a2.abs_charge == 3

def test_zero_neutral_mass():
    with pytest.raises(ValueError, match="must be positive"):
        calculate_precursor_mz(0.0, "[M+H]+")

def test_very_large_neutral_mass():
    mz = calculate_precursor_mz(1000000.0, "[M+H]+")
    mass_calc = calculate_neutral_mass(mz, "[M+H]+")
    np.testing.assert_allclose(mass_calc, 1000000.0, rtol=1e-6)

def test_get_adduct_candidates_invalid_mode():
    with pytest.raises(ValueError, match="Invalid ionization mode"):
        get_adduct_candidates(200.0, "neutral")

def test_calculate_neutral_mass_invalid_adduct():
    with pytest.raises(ValueError, match="Unknown or unsupported adduct"):
        calculate_neutral_mass(200.0, "[M+X]+")

def test_calculate_precursor_mz_invalid_adduct():
    with pytest.raises(ValueError, match="Unknown or unsupported adduct"):
        calculate_precursor_mz(200.0, "[M+X]+")

def test_detect_ms2_precursor_cluster_wrong_shape():
    assert not detect_ms2_precursor_cluster(200.0, np.array([1, 2, 3]))
    assert not detect_ms2_precursor_cluster(200.0, np.array([[1], [2]]))

def test_get_multihypothesis_precursor_candidates_mw_estimate():
    # base mass < 400, but mw_estimate >= 400
    res = get_multihypothesis_precursor_candidates(200.0, "[M+H]+", mw_estimate=500.0)
    assert len(res) == 3
    # base mass > 400, but mw_estimate < 400
    res2 = get_multihypothesis_precursor_candidates(500.0, "[M+H]+", mw_estimate=200.0)
    assert len(res2) == 2

def test_calculate_canonical_neutral_mass_both_formula_and_oxygens():
    # If both provided, explicit oxygens used
    calculate_canonical_neutral_mass(100.0, "[M-H2O+H]+", formula_oxygens=1, formula="C6H12") 
    with pytest.raises(ValueError, match="requires at least 1 oxygen"):
        calculate_canonical_neutral_mass(100.0, "[M-H2O+H]+", formula_oxygens=0, formula="C6H12O6")

def test_detect_ms2_precursor_cluster_zero_intensity():
    mz = 300.0
    peaks = np.array([
        [150.0, 0.0],
        [mz - CARBON_13_DELTA, 0.0]
    ])
    assert not detect_ms2_precursor_cluster(mz, peaks, min_intensity_ratio=0.1)
