"""
Unit tests for physical adduct plausibility and multi-isotopologue 13C deconvolution.
Task 1 of CASMI-Omega v2.
"""

from __future__ import annotations

import pytest
from src.chemistry.adducts import (
    CARBON_13_DELTA,
    calculate_canonical_neutral_mass,
    get_multihypothesis_precursor_candidates,
)


def test_water_loss_requires_oxygen_ge_2():
    # [M-2H2O+H]+ with formula having O=1 should raise ValueError
    with pytest.raises(ValueError, match="requires at least 2 oxygen atoms"):
        calculate_canonical_neutral_mass(
            mz=300.0, adduct="[M-2H2O+H]+", formula_oxygens=1
        )

    # Valid with O >= 2
    neutral_m = calculate_canonical_neutral_mass(
        mz=300.0, adduct="[M-2H2O+H]+", formula_oxygens=2
    )
    assert neutral_m > 300.0

    # Also test single water loss [M-H2O+H]+ with O < 1
    with pytest.raises(ValueError, match="requires at least 1 oxygen atom"):
        calculate_canonical_neutral_mass(
            mz=300.0, adduct="[M-H2O+H]+", formula_oxygens=0
        )

    # Valid single water loss with O >= 1
    neutral_m_single = calculate_canonical_neutral_mass(
        mz=300.0, adduct="[M-H2O+H]+", formula_oxygens=1
    )
    assert neutral_m_single > 300.0


def test_water_loss_via_formula_string():
    # Formula without oxygen should fail for [M-H2O+H]+
    with pytest.raises(ValueError, match="requires at least 1 oxygen atom"):
        calculate_canonical_neutral_mass(
            mz=300.0, adduct="[M-H2O+H]+", formula="C10H16"
        )

    # Formula with 1 oxygen should pass single water loss but fail double water loss
    neutral_m = calculate_canonical_neutral_mass(
        mz=300.0, adduct="[M-H2O+H]+", formula="C10H16O"
    )
    assert neutral_m > 0.0

    with pytest.raises(ValueError, match="requires at least 2 oxygen atoms"):
        calculate_canonical_neutral_mass(
            mz=300.0, adduct="[M-2H2O+H]+", formula="C10H16O"
        )

    # Formula with 2 oxygens passes double water loss
    neutral_m2 = calculate_canonical_neutral_mass(
        mz=300.0, adduct="[M-2H2O+H]+", formula="C10H16O2"
    )
    assert neutral_m2 > 300.0


def test_13c_multi_isotopologue_deconvolution():
    # Precursor at 505.0 Da with high MW should generate M, M-1.003355, M-2.006710
    hypotheses = get_multihypothesis_precursor_candidates(
        mz=505.0, adduct="[M+H]+", mw_estimate=504.0
    )
    assert len(hypotheses) == 3
    offsets = [h[1] for h in hypotheses]
    assert "M0" in offsets
    assert "M-1_13C" in offsets
    assert "M-2_13C" in offsets

    # Check masses and weights
    m0_mass, _, m0_weight = hypotheses[0]
    m1_mass, _, m1_weight = hypotheses[1]
    m2_mass, _, m2_weight = hypotheses[2]

    assert m0_weight == 1.0
    assert m1_weight == 0.35
    assert m2_weight == 0.08

    assert pytest.approx(m1_mass, abs=1e-5) == m0_mass - CARBON_13_DELTA
    assert pytest.approx(m2_mass, abs=1e-5) == m0_mass - 2.0 * CARBON_13_DELTA

    # For lower MW, M-2_13C should not be generated
    hypotheses_low = get_multihypothesis_precursor_candidates(
        mz=205.0, adduct="[M+H]+", mw_estimate=204.0
    )
    assert len(hypotheses_low) == 2
    offsets_low = [h[1] for h in hypotheses_low]
    assert "M0" in offsets_low
    assert "M-1_13C" in offsets_low
    assert "M-2_13C" not in offsets_low
