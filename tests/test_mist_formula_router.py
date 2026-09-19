import pytest
from src.retrieval.mist_formula_router import MISTFormulaRouter, FormulaRoutingDecision

def test_formula_neighborhood_expansion():
    router = MISTFormulaRouter(entropy_threshold=1.2)
    # Mock predicted distribution: C15H10O5 (prob 0.6), C15H12O5 (prob 0.3), C14H8O5 (prob 0.1)
    candidates = [("C15H10O5", 0.6), ("C15H12O5", 0.3), ("C14H8O5", 0.1)]

    decision = router.route_formula_distribution(candidates)
    assert not decision.use_formula_free_fallback
    assert "C15H10O5" in decision.formulas
    # Check neighborhood expansion (+1H, -1H, +1O, -1O)
    assert "C15H11O5" in decision.formulas
    assert "C15H9O5" in decision.formulas
    assert "C15H10O6" in decision.formulas
    assert "C15H10O4" in decision.formulas
    assert isinstance(decision.entropy, float)

def test_high_entropy_triggers_fallback():
    router = MISTFormulaRouter(entropy_threshold=1.0)
    # Highly dispersed predictions (5 candidates with 0.2 each, entropy = ln(5) ~= 1.609)
    dispersed = [(f"C{10+i}H{10+i}O3", 0.2) for i in range(5)]
    decision = router.route_formula_distribution(dispersed)
    assert decision.use_formula_free_fallback
    assert decision.entropy > 1.0

def test_single_candidate_formula():
    router = MISTFormulaRouter(entropy_threshold=1.2)
    candidates = [("C6H12O6", 1.0)]
    decision = router.route_formula_distribution(candidates)
    assert not decision.use_formula_free_fallback
    assert decision.entropy == 0.0
    assert "C6H12O6" in decision.formulas
    assert "C6H13O6" in decision.formulas
    assert "C6H11O6" in decision.formulas
    assert "C6H12O7" in decision.formulas
    assert "C6H12O5" in decision.formulas

def test_empty_candidate_list():
    router = MISTFormulaRouter(entropy_threshold=1.2)
    decision = router.route_formula_distribution([])
    assert decision.use_formula_free_fallback
    assert decision.entropy == 0.0
    assert decision.formulas == []

def test_formula_parsing_resilience():
    router = MISTFormulaRouter()
    # Empty string should return empty set or handled without error
    empty_exp = router.expand_neighborhood("")
    assert isinstance(empty_exp, set)

    # Formula without digits (counts = 1)
    ch4_exp = router.expand_neighborhood("CH4")
    assert "CH4" in ch4_exp
    assert "CH3" in ch4_exp
    assert "CH5" in ch4_exp
    assert "CH4O" in ch4_exp

    # Formula without H
    co2_exp = router.expand_neighborhood("CO2")
    assert "CO2" in co2_exp
    assert "CHO2" in co2_exp
    assert "CO3" in co2_exp
    assert "CO" in co2_exp
