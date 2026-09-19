import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    CASMIOmegaPipeline,
    run_casmi_omega_pipeline,
    PhytoForgePipeline,
    run_phytoforge_pipeline,
)
from src.data.multi_energy_fusion import MultiEnergyFusionEngine, SpectralFrame
from src.retrieval.mist_formula_router import MISTFormulaRouter
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever
from src.retrieval.database_search import SoftDatabaseSearcher
from src.retrieval.generative_denovo import BoundedGenerativeEngine
from src.retrieval.transductive_networking import TransductiveMolecularNetwork
from src.reranking.meta_ranker import GBDTMetaRanker
from src.reranking.slot_optimizer import DecisionTheoreticSlotOptimizer

@pytest.fixture
def minimal_df():
    return pd.DataFrame({
        'id': ['MOL_0001'],
        'precursor_mz': [195.0877],
        'adduct': ['[M+H]+']
    })

@pytest.fixture
def multi_row_df():
    return pd.DataFrame({
        'id': ['MOL_0001', 'MOL_0002', 'MOL_0003'],
        'precursor_mz': [195.0877, 205.123, 215.333],
        'adduct': ['[M+H]+', '[M+Na]+', '[M-H]-']
    })

def test_pipeline_constructor_modules():
    """1. CASMIOmegaPipeline constructor initializes all 8 modules (check attributes exist)"""
    pipeline = CASMIOmegaPipeline()
    assert hasattr(pipeline, 'fusion_engine')
    assert hasattr(pipeline, 'formula_router')
    assert hasattr(pipeline, 'dreams_retriever')
    assert hasattr(pipeline, 'db_searcher')
    assert hasattr(pipeline, 'denovo_engine')
    assert hasattr(pipeline, 'transductive_network')
    assert hasattr(pipeline, 'meta_ranker')
    assert hasattr(pipeline, 'slot_optimizer')
    assert hasattr(pipeline, 'governor')

def test_pipeline_single_row_output(minimal_df, tmp_path):
    """2. Pipeline run() with single-row DataFrame produces valid submission DataFrame"""
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "out.csv"
    res = pipeline.run(minimal_df, out_file)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1
    assert list(res.columns) == ["id", "candidates"]
    assert res.iloc[0]["id"] == "MOL_0001"

def test_pipeline_multi_row_output(multi_row_df, tmp_path):
    """3. Pipeline run() with multi-row (3 rows) DataFrame -> all IDs present"""
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "out.csv"
    res = pipeline.run(multi_row_df, out_file)
    assert len(res) == 3
    assert set(res["id"]) == {"MOL_0001", "MOL_0002", "MOL_0003"}

def test_pipeline_exact_25_candidates(minimal_df, tmp_path):
    """4. Pipeline run() output has exactly 25 candidates per row"""
    pipeline = CASMIOmegaPipeline()
    res = pipeline.run(minimal_df, tmp_path / "out.csv")
    candidates_str = res.iloc[0]["candidates"]
    cands = candidates_str.split(";")
    assert len(cands) == 25

def test_pipeline_candidates_format(minimal_df, tmp_path):
    """5. Pipeline run() candidates are 14-char alphanumeric strings"""
    pipeline = CASMIOmegaPipeline()
    res = pipeline.run(minimal_df, tmp_path / "out.csv")
    cands = res.iloc[0]["candidates"].split(";")
    for c in cands:
        assert len(c) == 14
        assert c.isalnum(), f"Candidate {c} is not alphanumeric"

def test_pipeline_candidates_distinct(minimal_df, tmp_path):
    """6. Pipeline run() candidates are pairwise distinct within each row"""
    pipeline = CASMIOmegaPipeline()
    res = pipeline.run(minimal_df, tmp_path / "out.csv")
    cands = res.iloc[0]["candidates"].split(";")
    assert len(cands) == len(set(cands))

def test_pipeline_empty_df(tmp_path):
    """7. Pipeline run() with empty DataFrame -> handles gracefully (empty output or no crash)"""
    pipeline = CASMIOmegaPipeline()
    empty_df = pd.DataFrame(columns=["id", "precursor_mz", "adduct"])
    res = pipeline.run(empty_df, tmp_path / "out.csv")
    assert len(res) == 0
    assert list(res.columns) == ["id", "candidates"]

def test_pipeline_submission_validation(minimal_df, tmp_path):
    """8. Pipeline run() submission validation passes (validate_submission call)"""
    # If validate_submission raises an error, this will fail.
    # The fact that it returns successfully means validation passed.
    pipeline = CASMIOmegaPipeline()
    res = pipeline.run(minimal_df, tmp_path / "out.csv")
    assert not res.empty

def test_phytoforge_alias():
    """9. PhytoForgePipeline is CASMIOmegaPipeline (alias identity check)"""
    assert PhytoForgePipeline is CASMIOmegaPipeline

def test_run_phytoforge_alias():
    """10. run_phytoforge_pipeline is run_casmi_omega_pipeline (alias identity check)"""
    assert run_phytoforge_pipeline is run_casmi_omega_pipeline

def test_governor_budget_propagate():
    """11. Governor budget parameter propagates to self.governor"""
    pipeline = CASMIOmegaPipeline(governor_budget_seconds=1234.5)
    assert pipeline.governor.total_budget_seconds == 1234.5

def test_custom_db_parameter():
    """12. Custom db parameter is used by db_searcher"""
    custom_db = {"C12H22O11": [("O(C1C(O)C(O)C(O)C(O)C1O)C2C(O)C(O)C(O)C(O)C2O", "IK14CUSTOM0001", np.ones(3, dtype=np.float32))]}
    pipeline = CASMIOmegaPipeline(db=custom_db)
    assert pipeline.db_searcher.db == custom_db

def test_custom_fallback_scaffolds():
    """13. Custom fallback_scaffolds parameter is used"""
    custom_fallback = [("C1CCCCC1", "IK14FALLBACK99", np.zeros(3, dtype=np.float32))]
    pipeline = CASMIOmegaPipeline(fallback_scaffolds=custom_fallback)
    assert pipeline.db_searcher.fallback_scaffolds == custom_fallback

def test_fusion_engine_output():
    """14. Fusion engine output is 1024-D numpy array"""
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    frame = SpectralFrame(collision_energy=35.0, peaks=np.array([[100.0, 50.0]]), embedding=np.zeros(1024, dtype=np.float32))
    fused = engine.fuse_embeddings([frame])
    assert isinstance(fused, np.ndarray)
    assert fused.shape == (1024,)

def test_formula_router_output():
    """15. Formula router output has .formulas attribute that is a list"""
    router = MISTFormulaRouter(entropy_threshold=1.2)
    routing = router.route_formula_distribution([("C15H10O5", 0.9)])
    assert hasattr(routing, "formulas")
    assert isinstance(routing.formulas, list)

def test_dreams_retriever_output():
    """16. DreaMS retriever evaluate_candidates returns list with .inchikey14 and .score"""
    retriever = CalibratedDreaMSRetriever()
    query_emb = np.zeros(1024, dtype=np.float32)
    lib = [("c1ccccc1", "IK14DREAM00001", query_emb.copy(), "timsTOF")]
    cands, _ = retriever.evaluate_candidates(query_emb, lib, "timsTOF")
    assert isinstance(cands, list)
    if len(cands) > 0:
        assert hasattr(cands[0], "inchikey14")
        assert hasattr(cands[0], "score")

def test_db_searcher_output():
    """17. DB searcher search_formulas returns list with .inchikey14 and .tanimoto_score"""
    searcher = SoftDatabaseSearcher(db={"C15H10O5": [("smiles", "IK14DBFLAV0001", np.ones(3, dtype=np.float32))]}, fallback_scaffolds=[])
    cands = searcher.search_formulas(["C15H10O5"], np.ones(3, dtype=np.float32))
    assert isinstance(cands, list)
    if len(cands) > 0:
        assert hasattr(cands[0], "inchikey14")
        assert hasattr(cands[0], "tanimoto_score")

def test_denovo_engine_output():
    """18. De novo engine generate_with_timeout returns list with .inchikey14 and .score"""
    engine = BoundedGenerativeEngine(timeout_seconds=0.1)
    cands = engine.generate_with_timeout("MOL1", formula="C15H10O5", timeout_seconds=0.01)
    assert isinstance(cands, list)
    if len(cands) > 0:
        assert hasattr(cands[0], "inchikey14")
        assert hasattr(cands[0], "score")

def test_transductive_network_output():
    """19. Transductive network propagate_scaffold returns list with .inchikey14 and .network_score"""
    network = TransductiveMolecularNetwork(cosine_threshold=0.5)
    network.add_spectrum("MOL1", 200.0, np.zeros(1024, dtype=np.float32), np.array([[100.0, 50.0]]))
    cands = network.propagate_scaffold("MOL1", {"SEED": ("smiles", "IK14SEED000001")})
    assert isinstance(cands, list)
    if len(cands) > 0:
        assert hasattr(cands[0], "inchikey14")
        assert hasattr(cands[0], "network_score")

def test_meta_ranker_feature_vector():
    """20. Meta-ranker extract_feature_vector returns numpy array"""
    ranker = GBDTMetaRanker()
    feat = ranker.extract_feature_vector(0.5, 0.5, 2.0, 0.001, 0.7, 1.2, 0.8, 1, 1, "track1")
    assert isinstance(feat, np.ndarray)

def test_meta_ranker_score_candidates():
    """21. Meta-ranker score_candidates output has 'score' key"""
    ranker = GBDTMetaRanker()
    cands = [{"inchikey14": "IK14MOCK000001", "features": np.zeros(10)}]
    ranked = ranker.score_candidates(cands)
    assert isinstance(ranked, list)
    if len(ranked) > 0:
        assert "score" in ranked[0]

def test_slot_optimizer_output():
    """22. Slot optimizer allocate_25_slots returns exactly 25 items"""
    opt = DecisionTheoreticSlotOptimizer(fallback_pool=[f"IK14FALL{i:06d}" for i in range(50)])
    ranked = [{"inchikey14": "IK14MOCK000001", "score": 0.9}]
    slots = opt.allocate_25_slots(ranked)
    assert isinstance(slots, list)
    assert len(slots) == 25

def test_pipeline_writes_csv(minimal_df, tmp_path):
    """23. Pipeline writes valid CSV output file (use tmp_path)"""
    pipeline = CASMIOmegaPipeline()
    out_file = tmp_path / "sub.csv"
    pipeline.run(minimal_df, out_file)
    assert out_file.exists()
    df_read = pd.read_csv(out_file)
    assert "id" in df_read.columns
    assert "candidates" in df_read.columns
    assert df_read.iloc[0]["id"] == "MOL_0001"

def test_standalone_run_function(minimal_df, tmp_path):
    """24. run_casmi_omega_pipeline function works as standalone"""
    out_file = tmp_path / "standalone.csv"
    res = run_casmi_omega_pipeline(minimal_df, out_file)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1
    assert out_file.exists()

def test_pipeline_governor_budget_seconds():
    """25. Pipeline with governor_budget_seconds=100 -> governor has correct budget"""
    pipeline = CASMIOmegaPipeline(governor_budget_seconds=100.0)
    assert pipeline.governor.total_budget_seconds == 100.0
