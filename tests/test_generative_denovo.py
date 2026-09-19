import time
import pytest
from src.retrieval.generative_denovo import BoundedGenerativeEngine, GenerativeCandidate

def test_hard_timeout_graceful_exit():
    def slow_generator(*args, **kwargs):
        time.sleep(0.5)
        return [("C1CC1", "IK14_SLOW", 0.5)]

    engine = BoundedGenerativeEngine(generator_fn=slow_generator, timeout_seconds=0.1)
    results = engine.generate_with_timeout(spectrum_id="S001", formula="C3H6")
    # Must exit gracefully on timeout without throwing
    assert results == []

def test_generation_bounded_candidates():
    def normal_generator(*args, **kwargs):
        return [(f"C{i}H{i*2}", f"IK14_{i}", 0.8) for i in range(10)]

    engine = BoundedGenerativeEngine(generator_fn=normal_generator, timeout_seconds=2.0, max_candidates=5)
    results = engine.generate_with_timeout(spectrum_id="S002", formula="C3H6")
    # Must cap at max 5 candidates
    assert len(results) <= 5
    assert len(results) == 5
    assert results[0].smiles == "C0H0"
    assert results[0].source_tier == "track3_denovo"

def test_generator_none_returns_empty():
    engine = BoundedGenerativeEngine(generator_fn=None)
    results = engine.generate_with_timeout(spectrum_id="S003", formula="C3H6")
    assert results == []

def test_generator_exception_handled_gracefully():
    def failing_generator(*args, **kwargs):
        raise RuntimeError("Model execution crashed")

    engine = BoundedGenerativeEngine(generator_fn=failing_generator, timeout_seconds=2.0)
    results = engine.generate_with_timeout(spectrum_id="S004", formula="C3H6")
    assert results == []

def test_generator_receives_max_steps():
    captured_kwargs = {}

    def inspect_generator(spectrum_id, formula, **kwargs):
        captured_kwargs.update(kwargs)
        return [("CC(=O)O", "QTBSBXVTEAMEQO", 0.95)]

    engine = BoundedGenerativeEngine(generator_fn=inspect_generator, max_steps=75)
    results = engine.generate_with_timeout(spectrum_id="S005", formula="C2H4O2")
    assert len(results) == 1
    assert captured_kwargs.get("max_steps") == 75

def test_generative_candidate_fields():
    cand = GenerativeCandidate(smiles="CCO", inchikey14="LFQSCWFLJHTTHZ", score=0.88)
    assert cand.smiles == "CCO"
    assert cand.inchikey14 == "LFQSCWFLJHTTHZ"
    assert cand.score == 0.88
    assert cand.source_tier == "track3_denovo"
