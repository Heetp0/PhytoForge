import numpy as np
import pytest
from src.chemistry.adducts import detect_halogen_presence_from_isotope_cluster
from src.retrieval.mist_formula_router import ExactMassFormulaIndex


class TestHalogenFilter:
    def test_detect_halogen_pure_organic(self):
        # Depsidone-like isotope pattern (no Br, no Cl)
        prec_mz = 375.0709
        peaks = np.array([
            [375.0709, 1.0000],
            [376.0744, 0.1449],
            [377.0694, 0.0482],
        ], dtype=np.float64)
        has_cl, has_br, m2_ratio = detect_halogen_presence_from_isotope_cluster(prec_mz, peaks)
        assert has_cl is False
        assert has_br is False
        assert 0.04 <= m2_ratio <= 0.05

    def test_detect_halogen_chlorine(self):
        # Monochlorinated pattern: [M+2] ~ 32%
        prec_mz = 300.0500
        peaks = np.array([
            [300.0500, 1.0000],
            [301.0530, 0.1200],
            [302.0470, 0.3250],
        ], dtype=np.float64)
        has_cl, has_br, m2_ratio = detect_halogen_presence_from_isotope_cluster(prec_mz, peaks)
        assert has_cl is True
        assert has_br is False
        assert m2_ratio >= 0.22

    def test_detect_halogen_bromine(self):
        # Monobrominated pattern: [M+2] ~ 97%
        prec_mz = 350.0200
        peaks = np.array([
            [350.0200, 1.0000],
            [351.0230, 0.1500],
            [352.0180, 0.9800],
        ], dtype=np.float64)
        has_cl, has_br, m2_ratio = detect_halogen_presence_from_isotope_cluster(prec_mz, peaks)
        assert has_cl is False
        assert has_br is True
        assert m2_ratio >= 0.70

    def test_find_formulas_halogen_exclusion(self):
        formulas = [
            "C18H14O9",      # 374.0638 (Pure NP)
            "C18H19BrN2O2",  # 374.0630 (Synthetic Br)
            "C16H14ClF3N2O3",# 374.0645 (Synthetic Cl/F)
        ]
        index = ExactMassFormulaIndex(formulas=formulas)
        target_mass = 374.0636

        # Exclude Br and Cl
        matched = index.find_formulas_by_mass(
            target_mass=target_mass,
            tolerance_ppm=20.0,
            has_chlorine=False,
            has_bromine=False,
        )
        matched_names = [f for f, _ in matched]
        assert "C18H14O9" in matched_names
        assert "C18H19BrN2O2" not in matched_names
        assert "C16H14ClF3N2O3" not in matched_names

    def test_find_formulas_fallback_if_all_excluded(self):
        # If database only had halogenated formulas, fallback to avoid returning empty
        formulas = ["C18H19BrN2O2"]
        index = ExactMassFormulaIndex(formulas=formulas)
        target_mass = 374.0636
        matched = index.find_formulas_by_mass(
            target_mass=target_mass,
            tolerance_ppm=20.0,
            has_chlorine=False,
            has_bromine=False,
        )
        assert len(matched) == 1
        assert matched[0][0] == "C18H19BrN2O2"
