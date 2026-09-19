import pytest
import numpy as np
import time
from typing import List

from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever, RetrievedCandidate
from src.retrieval.database_search import SoftDatabaseSearcher, DBCandidate
from src.retrieval.generative_denovo import BoundedGenerativeEngine, GenerativeCandidate
from src.retrieval.mist_formula_router import MISTFormulaRouter, FormulaRoutingDecision
from src.retrieval.transductive_networking import TransductiveMolecularNetwork, NetworkCandidate

# ---------------------------------------------------------------------------
# CalibratedDreaMSRetriever Tests
# ---------------------------------------------------------------------------

def test_dreams_empty_candidates():
    retriever = CalibratedDreaMSRetriever()
    res, locked = retriever.evaluate_candidates(np.array([1.0, 0.0]), [])
    assert res == []
    assert not locked

def test_dreams_below_threshold():
    retriever = CalibratedDreaMSRetriever()
    q_emb = np.array([1.0, 0.0])
    lib_emb = np.array([0.0, 1.0])
    cands = [("smi", "ik14", lib_emb, "timsTOF")]
    res, locked = retriever.evaluate_candidates(q_emb, cands)
    assert len(res) == 1
    assert res[0].score == 0.0
    assert not locked

def test_dreams_instrument_stratification():
    retriever = CalibratedDreaMSRetriever()
    # timsTOF -> generic threshold is generic (0.85). timsTOF->timsTOF is 0.88.
    q_emb = np.array([1.0, 0.0, 0.0])
    # 0.86 cosine sim
    lib_emb = np.array([0.86, 0.51, 0.0])
    
    # 0.86 > 0.85 (generic), but < 0.88 (timsTOF)
    cands_tims = [("smi", "ik", lib_emb, "timsTOF")]
    cands_gen = [("smi", "ik", lib_emb, "generic")]
    
    res_tims, locked_tims = retriever.evaluate_candidates(q_emb, cands_tims, "timsTOF")
    assert len(res_tims) == 1
    assert not locked_tims
    
    res_gen, locked_gen = retriever.evaluate_candidates(q_emb, cands_gen, "generic")
    assert len(res_gen) == 1
    assert locked_gen is True
    assert res_gen[0].score == pytest.approx(0.86, 0.05)

def test_dreams_all_zeros_query():
    retriever = CalibratedDreaMSRetriever()
    q_emb = np.zeros(5)
    lib_emb = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    res, locked = retriever.evaluate_candidates(q_emb, [("s", "i", lib_emb, "generic")])
    assert len(res) == 1
    assert res[0].score == 0.0
    assert not locked

def test_dreams_score_ordering():
    retriever = CalibratedDreaMSRetriever()
    q_emb = np.array([1.0, 0.0, 0.0])
    cands = [
        ("s1", "i1", np.array([0.9, 0.4, 0.0]), "generic"), # ~0.9
        ("s2", "i2", np.array([0.99, 0.1, 0.0]), "generic"), # ~0.99
    ]
    res, _ = retriever.evaluate_candidates(q_emb, cands, "generic")
    assert len(res) == 2
    assert res[0].inchikey14 == "i2"
    assert res[1].inchikey14 == "i1"

def test_dreams_fp16_inputs():
    retriever = CalibratedDreaMSRetriever()
    q_emb = np.array([1.0, 0.0], dtype=np.float16)
    lib_emb = np.array([1.0, 0.0], dtype=np.float16)
    res, _ = retriever.evaluate_candidates(q_emb, [("s", "i", lib_emb, "generic")], "generic")
    assert len(res) == 1
    assert res[0].score == pytest.approx(1.0, 1e-3)

def test_dreams_calibration_thresholds():
    retriever = CalibratedDreaMSRetriever()
    assert retriever.get_calibration_threshold("timsTOF", "Orbitrap") == 0.82
    assert retriever.get_calibration_threshold("unknown", "unknown") == 0.85

def test_dreams_zero_norm_candidate():
    retriever = CalibratedDreaMSRetriever()
    q_emb = np.array([1.0, 0.0])
    lib_emb = np.zeros(2)
    res, locked = retriever.evaluate_candidates(q_emb, [("s", "i", lib_emb, "generic")])
    assert len(res) == 1
    assert res[0].score == 0.0
    assert not locked

# ---------------------------------------------------------------------------
# SoftDatabaseSearcher Tests
# ---------------------------------------------------------------------------

def test_db_empty():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])
    res = searcher.search_formulas(["C6H12O6"], np.array([1.0, 1.0]))
    assert res == []

def test_db_empty_formulas():
    db = {"C6H12O6": [("s", "i", np.array([1.0, 1.0]))]}
    searcher = SoftDatabaseSearcher(db=db, fallback_scaffolds=[])
    res = searcher.search_formulas([], np.array([1.0, 1.0]))
    assert res == []

def test_db_formula_not_found():
    db = {"C6H12O6": [("s", "i", np.array([1.0, 1.0]))]}
    searcher = SoftDatabaseSearcher(db=db, fallback_scaffolds=[])
    res = searcher.search_formulas(["H2O"], np.array([1.0, 1.0]))
    assert res == []

def test_db_tanimoto_bounds():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])
    fp1 = np.array([1.0, 0.0, 1.0])
    fp2 = np.array([0.0, 1.0, 0.0])
    assert searcher.tanimoto(fp1, fp2) == 0.0
    assert searcher.tanimoto(fp1, fp1) == pytest.approx(1.0, 1e-9)

def test_db_all_zeros():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])
    z = np.zeros(3)
    assert searcher.tanimoto(z, z) == 0.0

def test_db_deduplication():
    # Provide the same inchikey twice under the same formula
    db = {
        "C": [
            ("smi1", "ik_same", np.array([1, 1, 0])),
            ("smi2", "ik_same", np.array([1, 1, 0])),
        ]
    }
    searcher = SoftDatabaseSearcher(db=db, fallback_scaffolds=[])
    res = searcher.search_formulas(["C"], np.array([1, 1, 0]))
    assert len(res) == 1

def test_db_batch_tanimoto():
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])
    q = np.array([1, 1, 0, 0])
    db_fps = np.array([
        [1, 1, 0, 0], # 2/2 = 1.0
        [1, 0, 0, 0], # 1/2 = 0.5
        [0, 0, 1, 1], # 0/4 = 0.0
    ])
    scores = searcher.batch_tanimoto(q, db_fps)
    np.testing.assert_allclose(scores, [1.0, 0.5, 0.0], rtol=1e-5)

def test_db_fallback():
    fallback = [("f_smi", "f_ik", np.array([1, 0]))]
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=fallback)
    res = searcher.search_formulas(["C"], np.array([1, 1]))
    assert len(res) == 1
    assert res[0].source_tier == "track2_fallback"

# ---------------------------------------------------------------------------
# BoundedGenerativeEngine Tests
# ---------------------------------------------------------------------------

def test_gen_no_generator():
    engine = BoundedGenerativeEngine(generator_fn=None)
    assert engine.generate_with_timeout("spec1") == []

def test_gen_none_formula():
    def dummy_gen(spec, form, max_steps): return [("s", "i", 0.5)]
    engine = BoundedGenerativeEngine(generator_fn=dummy_gen)
    res = engine.generate_with_timeout("spec", None)
    assert len(res) == 1

def test_gen_max_candidates():
    def dummy_gen(spec, form, max_steps): 
        return [("s1", "i1", 0.9), ("s2", "i2", 0.8), ("s3", "i3", 0.7)]
    engine = BoundedGenerativeEngine(generator_fn=dummy_gen, max_candidates=2)
    res = engine.generate_with_timeout("spec")
    assert len(res) == 2

def test_gen_timeout_enforcement():
    def slow_gen(spec, form, max_steps):
        time.sleep(1.0)
        return [("s", "i", 0.9)]
    engine = BoundedGenerativeEngine(generator_fn=slow_gen, timeout_seconds=0.01)
    start = time.time()
    res = engine.generate_with_timeout("spec")
    end = time.time()
    assert res == []
    assert (end - start) < 0.5 # Should exit quickly

def test_gen_valid_generative_candidates():
    def gen(spec, form, max_steps):
        return [GenerativeCandidate("s", "i", 0.9)]
    engine = BoundedGenerativeEngine(generator_fn=gen)
    res = engine.generate_with_timeout("s")
    assert len(res) == 1
    assert isinstance(res[0], GenerativeCandidate)
    assert res[0].smiles == "s"

def test_gen_exception():
    def bad_gen(spec, form, max_steps):
        raise ValueError("Oops")
    engine = BoundedGenerativeEngine(generator_fn=bad_gen)
    res = engine.generate_with_timeout("s")
    assert res == []

def test_gen_preserve_spectrum_id():
    def check_spec_gen(spec, form, max_steps):
        return [(spec, "ik", 1.0)]
    engine = BoundedGenerativeEngine(generator_fn=check_spec_gen)
    res = engine.generate_with_timeout("MY_SPEC_ID")
    assert len(res) == 1
    assert res[0].smiles == "MY_SPEC_ID"

def test_gen_invalid_tuple_return():
    def bad_tuple_gen(spec, form, max_steps):
        return [("s",)] # Too short
    engine = BoundedGenerativeEngine(generator_fn=bad_tuple_gen)
    res = engine.generate_with_timeout("s")
    assert res == []

# ---------------------------------------------------------------------------
# MISTFormulaRouter Tests
# ---------------------------------------------------------------------------

def test_mist_empty_priors():
    router = MISTFormulaRouter()
    dec = router.route_formula_distribution([])
    assert dec.formulas == []
    assert dec.entropy == 0.0
    assert dec.use_formula_free_fallback

def test_mist_low_entropy():
    router = MISTFormulaRouter(entropy_threshold=1.5)
    dec = router.route_formula_distribution([("C6H12O6", 0.99), ("C5H10O5", 0.01)])
    assert dec.entropy < 1.5
    assert not dec.use_formula_free_fallback

def test_mist_high_entropy():
    router = MISTFormulaRouter(entropy_threshold=0.1)
    # Uniform over 3 gives ~1.09
    dec = router.route_formula_distribution([("C", 0.33), ("H", 0.33), ("O", 0.33)])
    assert dec.use_formula_free_fallback
    assert dec.entropy > 0.1

def test_mist_expand_neighborhood():
    router = MISTFormulaRouter()
    expanded = router.expand_neighborhood("C6H12O6")
    assert "C6H12O6" in expanded
    assert "C6H11O6" in expanded
    assert "C6H13O6" in expanded
    assert "C6H12O5" in expanded
    assert "C6H12O7" in expanded
    # Wait, the code says:
    # +/- 1H: cand = elements.copy(); cand['H'] += dh
    # +/- 1O: cand = elements.copy(); cand['O'] += do
    # They are independent modifications of the base formula.
    assert "C6H13O7" not in expanded # Correct, they are independently modified from base

def test_mist_shannon_single():
    router = MISTFormulaRouter()
    assert router.calculate_entropy([1.0]) == 0.0

def test_mist_shannon_zero_total():
    router = MISTFormulaRouter()
    assert router.calculate_entropy([0.0, 0.0]) == 0.0

def test_mist_no_carbon_expand():
    router = MISTFormulaRouter()
    expanded = router.expand_neighborhood("H2O")
    # Base is H2O
    assert "H2O" in expanded
    assert "H3O" in expanded
    assert "HO" in expanded
    assert "H2" in expanded
    assert "H2O2" in expanded

def test_mist_bad_formula_expand():
    router = MISTFormulaRouter()
    assert router.expand_neighborhood("") == set()
    assert router.expand_neighborhood("!@#$") == set()

# ---------------------------------------------------------------------------
# TransductiveMolecularNetwork Tests
# ---------------------------------------------------------------------------

def test_tmn_empty():
    tmn = TransductiveMolecularNetwork()
    assert tmn.propagate_scaffold("spec1", {}) == []

def test_tmn_single_node():
    tmn = TransductiveMolecularNetwork()
    tmn.add_spectrum("s1", 100.0, np.array([1, 1]), np.array([[10, 100]]))
    assert tmn.propagate_scaffold("s1", {"s1": ("s", "i")}) == []

def test_tmn_match_exact_delta():
    tmn = TransductiveMolecularNetwork(mass_tolerance_ppm=20.0)
    assert tmn.match_delta(162.0528) == "+Hexose"

def test_tmn_match_no_delta():
    tmn = TransductiveMolecularNetwork()
    assert tmn.match_delta(500.0) is None

def test_tmn_cosine_threshold():
    tmn = TransductiveMolecularNetwork(cosine_threshold=0.7)
    
    # Cosine = 0.6
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.6, 0.8]) 
    
    tmn.add_spectrum("t", 100.0, v1, np.array([[10, 100], [20, 100]]))
    tmn.add_spectrum("s", 100.0 + 162.0528, v2, np.array([[10, 100], [20, 100]]))
    
    res = tmn.propagate_scaffold("t", {"s": ("smi", "ik")})
    assert len(res) == 0 # Cosine < 0.7

def test_tmn_cosine_above_threshold():
    tmn = TransductiveMolecularNetwork(cosine_threshold=0.7)
    
    # Cosine = 1.0
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0]) 
    
    tmn.add_spectrum("t", 100.0, v1, np.array([[10, 100], [20, 100]]))
    tmn.add_spectrum("s", 100.0 + 162.0528, v2, np.array([[10, 100], [20, 100]]))
    
    res = tmn.propagate_scaffold("t", {"s": ("smi", "ik")})
    assert len(res) == 1
    assert res[0].transformation == "+Hexose"

def test_tmn_shared_peaks():
    tmn = TransductiveMolecularNetwork(cosine_threshold=0.0)
    v1 = np.array([1.0, 0.0])
    
    # Only 1 shared peak (10.0)
    tmn.add_spectrum("t", 100.0, v1, np.array([[10.0, 100], [20.0, 100]]))
    tmn.add_spectrum("s", 100.0 + 162.0528, v1, np.array([[10.0, 100], [30.0, 100]]))
    
    res = tmn.propagate_scaffold("t", {"s": ("smi", "ik")})
    assert len(res) == 0 # Needs >= 2

def test_tmn_score_monotonicity():
    tmn = TransductiveMolecularNetwork(cosine_threshold=0.0)
    v_t = np.array([1.0, 0.0])
    v_s1 = np.array([0.9, 0.4358]) # ~0.9 cosine
    v_s2 = np.array([0.99, 0.141]) # ~0.99 cosine
    
    tmn.add_spectrum("t", 100.0, v_t, np.array([[10, 100], [20, 100]]))
    tmn.add_spectrum("s1", 100.0 + 162.0528, v_s1, np.array([[10, 100], [20, 100]]))
    tmn.add_spectrum("s2", 100.0 + 162.0528, v_s2, np.array([[10, 100], [20, 100]]))
    
    res = tmn.propagate_scaffold("t", {"s1": ("smi1", "ik1"), "s2": ("smi2", "ik2")})
    assert len(res) == 2
    assert res[0].source_id == "s2"
    assert res[1].source_id == "s1"
    assert res[0].network_score > res[1].network_score
