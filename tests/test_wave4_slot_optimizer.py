"""Wave 4 Test Suite: Decision-Theoretic 25-Slot Allocator & Submission Integrity.

Verifies:
1. Canonical botanical aglycone diversity bank (InChIKey14 and SMILES).
2. Elimination of synthetic dummy strings (IK14FALL, PAD00000000000, methane/ethane).
3. Dynamic range-preserving MMR score normalization.
4. Robust 25-slot invariant preservation when salts/mixtures are stripped.
5. End-to-end atomic submission writer format compliance on both full and depleted candidate pools.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.reranking.slot_optimizer import (
    DecisionTheoreticSlotOptimizer,
    BOTANICAL_AGLYCONE_INCHIKEY14,
    BOTANICAL_AGLYCONE_SMILES,
)
from src.submission.writer import (
    validate_submission_file,
    validate_submission,
    write_submission,
    DEFAULT_FALLBACK_SMILES,
)


def test_botanical_aglycone_constants():
    """Verify botanical diversity constants are non-empty, strictly valid, and pairwise distinct."""
    assert len(BOTANICAL_AGLYCONE_INCHIKEY14) >= 50
    assert len(BOTANICAL_AGLYCONE_INCHIKEY14) == len(set(BOTANICAL_AGLYCONE_INCHIKEY14))
    for k in BOTANICAL_AGLYCONE_INCHIKEY14:
        assert isinstance(k, str)
        assert len(k) == 14
        assert k.isalnum()
        assert k.isupper()

    assert len(BOTANICAL_AGLYCONE_SMILES) >= 30
    assert len(BOTANICAL_AGLYCONE_SMILES) == len(set(BOTANICAL_AGLYCONE_SMILES))
    for s in BOTANICAL_AGLYCONE_SMILES:
        assert isinstance(s, str)
        assert "+" not in s
        assert not any(c in s for c in (" ", "\t", "\n", "\r", ","))


def test_slot_optimizer_zero_candidates_fallback():
    """Verify zero-candidate queries yield exactly 25 authentic botanical aglycones with zero dummies."""
    optimizer = DecisionTheoreticSlotOptimizer()
    slots = optimizer.allocate_25_slots([])

    assert len(slots) == 25
    assert len(set(slots)) == 25
    # Must not contain dummy strings
    assert not any("PAD" in s or "IK14FALL" in s for s in slots)
    # Must be drawn from the botanical diversity bank
    assert all(s in BOTANICAL_AGLYCONE_INCHIKEY14 for s in slots)


def test_slot_optimizer_partial_candidates_enrichment():
    """Verify partial candidate sets are backfilled to exactly 25 slots without duplicate scaffolds."""
    optimizer = DecisionTheoreticSlotOptimizer()
    cands = [
        {"inchikey14": "REFJWTPEDVJJIY", "score": 0.95, "smiles": "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13"}, # Quercetin
        {"inchikey14": "ISWSIDIOOBJBQZ", "score": 0.85, "smiles": "O=C1C(O)=C(c2ccc(O)cc2)Oc3cc(O)cc(O)c13"},     # Kaempferol
        {"inchikey14": "KZNIFHHRGQBSSA", "score": 0.75, "smiles": "O=C1C=C(c2ccc(O)cc2)Oc3cc(O)cc(O)c13"},         # Apigenin
    ]
    slots = optimizer.allocate_25_slots(cands, diversity_lambda=0.85)

    assert len(slots) == 25
    assert len(set(slots)) == 25
    # First 3 slots must be the candidate pool
    assert slots[0] == "REFJWTPEDVJJIY"
    assert slots[1] == "ISWSIDIOOBJBQZ"
    assert slots[2] == "KZNIFHHRGQBSSA"
    # Remaining 22 slots must be non-redundant botanical aglycones
    assert not any("PAD" in s or "IK14FALL" in s for s in slots)


def test_slot_optimizer_mmr_score_preservation():
    """Verify score normalization preserves discriminative ranking margins and MMR trade-off."""
    optimizer = DecisionTheoreticSlotOptimizer()
    # 35 candidates with linearly spaced scores in [0.05, 0.95]
    cands = [
        {"inchikey14": f"TESTCAND{i:06d}", "score": 0.05 + i * 0.025, "smiles": "c1ccccc1"}
        for i in range(35)
    ]
    slots = optimizer.allocate_25_slots(cands, diversity_lambda=0.85)

    assert len(slots) == 25
    assert len(set(slots)) == 25
    # Highest scoring candidate should be Rank 1
    assert slots[0] == "TESTCAND000034"
    assert slots[1] == "TESTCAND000033"
    assert slots[2] == "TESTCAND000032"


def test_write_submission_inchikey14_fallback_enrichment(tmp_path: Path):
    """Verify write_submission with inchikey14 format produces clean botanical fallback rows."""
    out_csv = tmp_path / "submission_ik14.csv"
    preds = {
        "SPEC_001": ["REFJWTPEDVJJIY", "ISWSIDIOOBJBQZ"], # Only 2 candidates
        "SPEC_002": [],                                      # 0 candidates
    }
    written_path = write_submission(
        predictions=preds,
        output_path=out_csv,
        expected_ids=["SPEC_001", "SPEC_002"],
        output_format="inchikey14",
        auto_backfill=True,
    )

    assert written_path.exists()
    is_valid, errors = validate_submission_file(written_path, expected_ids=["SPEC_001", "SPEC_002"], output_format="inchikey14")
    assert is_valid, f"Validation errors: {errors}"

    content = written_path.read_text(encoding="utf-8")
    assert "IK14FALL" not in content
    assert "PAD0000" not in content


def test_write_submission_smiles_fallback_enrichment(tmp_path: Path):
    """Verify write_submission with smiles format produces clean botanical SMILES rows."""
    out_csv = tmp_path / "submission_smiles.csv"
    preds = {
        "SPEC_001": ["O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13"], # 1 candidate
    }
    written_path = write_submission(
        predictions=preds,
        output_path=out_csv,
        expected_ids=["SPEC_001"],
        output_format="smiles",
        auto_backfill=True,
    )

    assert written_path.exists()
    is_valid, errors = validate_submission_file(written_path, expected_ids=["SPEC_001"], output_format="smiles")
    assert is_valid, f"Validation errors: {errors}"

    content = written_path.read_text(encoding="utf-8")
    # Verify no methane/ethane fallback
    assert ";C;" not in content
    assert ";CC;" not in content
