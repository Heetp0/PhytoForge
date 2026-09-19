import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.reranking.meta_ranker import GBDTMetaRanker
from src.reranking.slot_optimizer import DecisionTheoreticSlotOptimizer
from src.data.multi_energy_fusion import MultiEnergyFusionEngine, SpectralFrame
from src.retrieval.mist_formula_router import MISTFormulaRouter, FormulaRoutingDecision
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever, RetrievedCandidate
from src.retrieval.database_search import SoftDatabaseSearcher, DBCandidate
from src.retrieval.generative_denovo import BoundedGenerativeEngine, GenerativeCandidate
from src.retrieval.transductive_networking import TransductiveMolecularNetwork, NetworkCandidate
from src.submission.writer import DEFAULT_FALLBACK_SMILES, get_inchikey14, validate_submission
from src.pipeline import CASMIOmegaPipeline

def test_meta_ranker_output_keys():
    ranker = GBDTMetaRanker()
    cands = [{"features": np.zeros(32, dtype=np.float32)}]
    scored = ranker.score_candidates(cands)
    assert "score" in scored[0]
    assert "meta_score" in scored[0]
    assert isinstance(scored[0]["score"], float)

def test_slot_optimizer_extract_score_key():
    opt = DecisionTheoreticSlotOptimizer()
    slots = opt.allocate_25_slots([{"inchikey14": "12345678901234", "score": 0.9}])
    assert slots[0] == "12345678901234"

def test_slot_optimizer_extract_meta_score_key():
    opt = DecisionTheoreticSlotOptimizer()
    slots = opt.allocate_25_slots([{"inchikey14": "12345678901234", "meta_score": 0.9}])
    assert slots[0] == "12345678901234"

def test_slot_optimizer_extract_score_attribute():
    class Cand:
        def __init__(self, inchikey14, score):
            self.inchikey14 = inchikey14
            self.score = score
    opt = DecisionTheoreticSlotOptimizer()
    slots = opt.allocate_25_slots([Cand("12345678901234", 0.9)])
    assert slots[0] == "12345678901234"

def test_fusion_engine_output_shape():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    frame = SpectralFrame(embedding=np.ones(1024, dtype=np.float32))
    fused = engine.fuse_embeddings([frame])
    assert fused.shape == (1024,)
    assert fused.dtype == np.float32

def test_formula_router_output_type():
    router = MISTFormulaRouter()
    decision = router.route_formula_distribution([("C15H10O5", 0.9)])
    assert isinstance(decision, FormulaRoutingDecision)
    assert isinstance(decision.formulas, list)
    if decision.formulas:
        assert isinstance(decision.formulas[0], str)

def test_dreams_retriever_candidate_attrs():
    retriever = CalibratedDreaMSRetriever()
    query = np.ones(10, dtype=np.float32)
    lib = [("smiles", "12345678901234", np.ones(10, dtype=np.float32), "timsTOF")]
    cands, _ = retriever.evaluate_candidates(query, lib)
    assert len(cands) == 1
    assert hasattr(cands[0], "inchikey14")
    assert hasattr(cands[0], "score")

def test_db_searcher_candidate_attrs():
    db = {"C": [("C", "12345678901234", np.ones(3, dtype=np.float32))]}
    searcher = SoftDatabaseSearcher(db=db, fallback_scaffolds=[])
    cands = searcher.search_formulas(["C"], np.ones(3, dtype=np.float32))
    assert len(cands) == 1
    assert hasattr(cands[0], "inchikey14")
    assert hasattr(cands[0], "tanimoto_score")

def test_generative_denovo_candidate_attrs():
    def dummy_gen(spec_id, form, max_steps):
        return [GenerativeCandidate("C", "12345678901234", 0.9)]
    engine = BoundedGenerativeEngine(generator_fn=dummy_gen)
    cands = engine.generate_with_timeout("spec1")
    assert len(cands) == 1
    assert hasattr(cands[0], "inchikey14")
    assert hasattr(cands[0], "score")

def test_transductive_networking_candidate_attrs():
    network = TransductiveMolecularNetwork(cosine_threshold=0.0)
    network.add_spectrum("t1", 100.0, np.ones(10), np.array([[100.0, 50.0], [150.0, 50.0]]))
    network.add_spectrum("s1", 100.0 + 14.0156, np.ones(10), np.array([[100.0, 50.0], [150.0, 50.0]]))
    cands = network.propagate_scaffold("t1", {"s1": ("smiles", "12345678901234")})
    assert len(cands) == 1
    assert hasattr(cands[0], "inchikey14")
    assert hasattr(cands[0], "network_score")

def test_fallback_pool_inchikey14_validity():
    opt = DecisionTheoreticSlotOptimizer(fallback_pool=["VALIDKEY123456", "inv@lid", "short"])
    assert "VALIDKEY123456" in opt.fallback_pool
    assert "inv@lid" not in opt.fallback_pool
    assert "short" not in opt.fallback_pool

def test_default_fallback_smiles_validity():
    for smiles in DEFAULT_FALLBACK_SMILES:
        ik14 = get_inchikey14(smiles)
        assert ik14 is not None
        assert len(ik14) == 14
        assert ik14.isalnum()

def test_pipeline_candidate_aggregation_dict_keys(tmp_path):
    pipeline = CASMIOmegaPipeline()
    df = pd.DataFrame({"id": ["spec1"], "precursor_mz": [100.0], "adduct": ["[M+H]+"]})
    out = tmp_path / "sub.csv"
    res_df = pipeline.run(df, out)
    assert len(res_df) == 1
    assert "candidates" in res_df.columns
    cands = res_df.iloc[0]["candidates"].split(";")
    assert len(cands) == 25

def test_slot_optimizer_passes_validate_submission():
    opt = DecisionTheoreticSlotOptimizer()
    slots = opt.allocate_25_slots([{"inchikey14": "12345678901234", "score": 0.9}])
    df = pd.DataFrame({"id": ["spec1"], "candidates": [";".join(slots)]})
    assert validate_submission(df, ["spec1"]) is True

def test_gbdt_meta_ranker_feature_dim():
    ranker = GBDTMetaRanker()
    vec = ranker.extract_feature_vector(
        0.5, 0.5, 1.0, 0.01, 0.5, 0.5, 0.5, 1, 1, "track1_dreams"
    )
    assert vec.shape == (32,)
    assert ranker.weights.shape == (32,)

def test_formula_routing_decision_dataclass():
    d = FormulaRoutingDecision(formulas=["C"], entropy=0.5, use_formula_free_fallback=False)
    assert d.formulas == ["C"]
    assert d.entropy == 0.5
    assert d.use_formula_free_fallback is False

def test_retrieved_candidate_dataclass():
    c = RetrievedCandidate("smiles", "ik14", 0.9, "inst")
    assert c.smiles == "smiles"
    assert c.inchikey14 == "ik14"
    assert c.score == 0.9
    assert c.instrument == "inst"

def test_db_candidate_dataclass():
    c = DBCandidate("smiles", "ik14", 0.8)
    assert c.smiles == "smiles"
    assert c.inchikey14 == "ik14"
    assert c.tanimoto_score == 0.8

def test_generative_candidate_dataclass():
    c = GenerativeCandidate("smiles", "ik14", 0.7)
    assert c.smiles == "smiles"
    assert c.inchikey14 == "ik14"
    assert c.score == 0.7

def test_network_candidate_dataclass():
    c = NetworkCandidate("s1", "smiles", "ik14", "trans", 0.6)
    assert c.source_id == "s1"
    assert c.scaffold_smiles == "smiles"
    assert c.inchikey14 == "ik14"
    assert c.transformation == "trans"
    assert c.network_score == 0.6
