import pytest
from src.reranking.slot_optimizer import DecisionTheoreticSlotOptimizer


def test_mmr_slot_optimizer_diversifies_similar_candidates():
    """Verify that MMR penalizes redundant near-identical analogs and promotes a diverse candidate into top slots."""
    optimizer = DecisionTheoreticSlotOptimizer()

    # Create 5 candidates:
    # Cand 1, 2, 3: Flavonoids (Quercetin, Kaempferol, Apigenin) - very high similarity to each other (~0.78)
    # Cand 4: Berberine (Alkaloid) - very different (~0.07 similarity), slightly lower score
    candidates = [
        {"inchikey14": "IK14QUERCET001", "smiles": "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "score": 0.95},
        {"inchikey14": "IK14KAEMPFE002", "smiles": "O=C1C(O)=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "score": 0.94},
        {"inchikey14": "IK14APIGENI003", "smiles": "O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "score": 0.93},
        {"inchikey14": "IK14LUTEOLI004", "smiles": "O=C1C=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "score": 0.92},
        {"inchikey14": "IK14BERBERI005", "smiles": "COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2", "score": 0.90},
    ]

    # With greedy ranking (lambda = 1.0), order is 1, 2, 3, 4, 5
    greedy_slots = optimizer.allocate_25_slots(candidates, diversity_lambda=1.0)
    assert greedy_slots[0] == "IK14QUERCET001"
    assert greedy_slots[3] == "IK14LUTEOLI004"
    assert greedy_slots[4] == "IK14BERBERI005"

    # With MMR diversity (lambda = 0.70), slot 4 should prioritize the diverse Berberine over the 4th flavonoid analog
    mmr_slots = optimizer.allocate_25_slots(candidates, diversity_lambda=0.70)
    assert mmr_slots[0] == "IK14QUERCET001"  # Top candidate protected
    assert "IK14BERBERI005" in mmr_slots[:5]
    # Berberine should be promoted ahead of the 4th repetitive flavonoid
    assert mmr_slots.index("IK14BERBERI005") < mmr_slots.index("IK14LUTEOLI004")


def test_mmr_slot_optimizer_guarantees_25_unique_slots():
    """Verify that MMR slotting always returns strictly 25 unique valid 14-char alphanumeric keys."""
    optimizer = DecisionTheoreticSlotOptimizer()
    candidates = [
        {"inchikey14": f"IK14CAND{i:06d}", "smiles": "C" * (i + 1), "score": 1.0 / (i + 1)}
        for i in range(10)
    ]
    slots = optimizer.allocate_25_slots(candidates, diversity_lambda=0.88)
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert all(len(s) == 14 and s.isalnum() for s in slots)
