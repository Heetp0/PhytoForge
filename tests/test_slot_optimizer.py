import pytest
from src.reranking.slot_optimizer import DecisionTheoreticSlotOptimizer


def test_slot_optimizer_strict_25_uniques():
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=[f"IK14FALL{i:06d}" for i in range(50)])

    # Input with duplicates and stereoisomers sharing InChIKey14
    raw = [
        {"inchikey14": "IK14AAAA000001", "score": 0.95, "source": "track1"},
        {"inchikey14": "IK14AAAA000001", "score": 0.90, "source": "track2"},  # Duplicate
        {"inchikey14": "IK14BBBB000002", "score": 0.85, "source": "track1"},
    ]
    slots = optimizer.allocate_25_slots(raw)
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert slots[0] == "IK14AAAA000001"
    assert slots[1] == "IK14BBBB000002"
    assert slots[2] == "IK14FALL000000"  # Padded from fallback
    assert all(len(s) == 14 and s.isalnum() for s in slots)


def test_slot_optimizer_empty_candidates_fallback_padding():
    fallback = [f"FALLBACKKEY{i:03d}" for i in range(30)]
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=fallback)

    slots = optimizer.allocate_25_slots([])
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert slots == fallback[:25]
    assert all(len(s) == 14 and s.isalnum() for s in slots)


def test_slot_optimizer_empty_candidates_emergency_padding():
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=[])

    slots = optimizer.allocate_25_slots([])
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert slots[0] == "PAD00000000000"
    assert slots[24] == "PAD00000000024"
    assert all(len(s) == 14 and s.isalnum() for s in slots)


def test_slot_optimizer_fallback_smaller_than_25():
    fallback = [f"SHORTFALL0000{i}" for i in range(5)]
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=fallback)

    # 2 valid candidates + 5 fallback = 7, remaining 18 padded from emergency pad
    raw = [
        {"inchikey14": "CAND14TEST0001", "score": 0.9},
        {"inchikey14": "CAND14TEST0002", "score": 0.8},
    ]
    slots = optimizer.allocate_25_slots(raw)
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert slots[0] == "CAND14TEST0001"
    assert slots[1] == "CAND14TEST0002"
    assert slots[2:7] == fallback
    assert slots[7] == "PAD00000000000"
    assert slots[24] == "PAD00000000017"
    assert all(len(s) == 14 and s.isalnum() for s in slots)


def test_slot_optimizer_invalid_format_filtering():
    # Both candidates and fallback pool contain malformed entries
    fallback = [
        "VALIDFALL00001",
        "TOO_SHORT",            # < 14 chars
        "THISKEYISTOOLONG12345", # > 14 chars
        "INCHI-KEY-HYPH",       # contains hyphens
        "SPECIAL!CHARS#",       # contains special symbols
        "VALIDFALL00002",
        "",                     # empty
    ]
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=fallback)

    raw = [
        {"inchikey14": "VALIDCAND00001", "score": 0.95},
        {"inchikey14": "INVALID-HYPHEN", "score": 0.90},       # Non-alphanumeric
        {"inchikey14": "SHORT", "score": 0.85},                # Too short
        {"inchikey14": "TOOLONGKEY1234567890", "score": 0.80}, # Too long
        {"inchikey14": None, "score": 0.75},                   # None
        {"inchikey14": "", "score": 0.70},                     # Empty string
        {"inchikey14": "VALIDCAND00002", "score": 0.65},
    ]

    slots = optimizer.allocate_25_slots(raw)
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert slots[0] == "VALIDCAND00001"
    assert slots[1] == "VALIDCAND00002"
    assert slots[2] == "VALIDFALL00001"
    assert slots[3] == "VALIDFALL00002"
    # Remaining 21 slots emergency padded
    assert all(len(s) == 14 and s.isalnum() for s in slots)


def test_slot_optimizer_candidate_scores_prioritization_order():
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=[f"FALLBACKKEY{i:03d}" for i in range(30)])

    # Candidates passed in non-sorted order by score
    raw = [
        {"inchikey14": "LOWSCORE000001", "score": 0.20},
        {"inchikey14": "HIGHSCORE00001", "score": 0.99},
        {"inchikey14": "MIDSCORE000001", "score": 0.60},
    ]
    slots = optimizer.allocate_25_slots(raw)
    assert slots[0] == "HIGHSCORE00001"
    assert slots[1] == "MIDSCORE000001"
    assert slots[2] == "LOWSCORE000001"
    assert len(slots) == 25
    assert len(set(slots)) == 25
    assert all(len(s) == 14 and s.isalnum() for s in slots)


def test_slot_optimizer_meta_score_prioritization_order():
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=[f"FALLBACKKEY{i:03d}" for i in range(30)])

    # Candidates output by GBDTMetaRanker with "meta_score" in arbitrary order
    raw = [
        {"inchikey14": "LOWMETA0000001", "meta_score": 0.15},
        {"inchikey14": "HIGHMETA000001", "meta_score": 0.98},
        {"inchikey14": "MIDMETA0000001", "meta_score": 0.55},
    ]
    slots = optimizer.allocate_25_slots(raw)
    assert slots[0] == "HIGHMETA000001"
    assert slots[1] == "MIDMETA0000001"
    assert slots[2] == "LOWMETA0000001"
