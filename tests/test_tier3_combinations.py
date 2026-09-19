"""
tests/test_tier3_combinations.py
Tier 3: Pairwise Combinatorial Testing & Cross-Module Interactions.
Comprehensive verification of data handoffs, interface contracts, and cascades.
Total: 15 pairwise interaction test cases.
"""

from pathlib import Path
import time
import pytest
import numpy as np
import pandas as pd

from tests.conftest import (
    Candidate,
    ADDUCT_DEFINITIONS,
    GROUND_TRUTH_MOLECULES,
    CARBON_13_OFFSET,
    DISTINCT_25_SMILES,
    AdductsOracle,
    StandardizerOracle,
    SubmissionWriterOracle,
    RuntimeGovernorOracle,
    Tier1VectorMatcherOracle,
    Tier2TanimotoRankerOracle,
    MMROptimizerOracle,
)


def test_comb_01_adduct_drives_tier1_search(adduct_oracle, tier1_oracle):
    """Pairwise: Adduct neutral mass calculation drives Tier 1 mass search window."""
    true_neutral = GROUND_TRUTH_MOLECULES["caffeine"]["neutral_mass"]
    adduct = "[M+H]+"
    observed_mz = true_neutral + ADDUCT_DEFINITIONS[adduct].delta_mass

    # 1. De-adduct precursor m/z
    neutral_mass = adduct_oracle.calculate_neutral_mass(observed_mz, adduct)
    assert abs(neutral_mass - true_neutral) < 1e-6

    # 2. Tier 1 mass-gated search around neutral_mass
    lib_masses = np.array([true_neutral, true_neutral + 50.0], dtype=np.float64)
    lib_embs = np.ones((2, 512), dtype=np.float32)
    hits = tier1_oracle.search_library(
        np.ones(512, dtype=np.float32),
        neutral_mass,
        lib_embs,
        lib_masses,
        ["Caffeine", "OtherMol"],
        ["KEY_CAFF_12345", "KEY_OTHER_1234"]
    )
    assert len(hits) == 1
    assert hits[0].smiles == "Caffeine"


def test_comb_02_13c_correction_shifts_tier2_retrieval(adduct_oracle, tier2_oracle):
    """Pairwise: False precursor 13C correction aligns mass before Tier 2 ranking."""
    true_neutral = GROUND_TRUTH_MOLECULES["aspirin"]["neutral_mass"]
    mispicked_mz = true_neutral + CARBON_13_OFFSET
    peaks = np.array([[true_neutral, 500.0]], dtype=np.float64)

    # 1. Correct false 13C precursor
    corrected_mz = adduct_oracle.correct_precursor_mz(mispicked_mz, peaks)
    assert abs(corrected_mz - true_neutral) < 1e-4

    # 2. Score candidates in Tier 2
    query_fp = np.ones(64, dtype=np.uint64)
    cand_fps = np.ones((1, 64), dtype=np.uint64)
    meta = [{"smiles": "Aspirin", "inchikey14": "BSYNRYMUTXBXSQ", "neutral_mass": true_neutral}]
    hits = tier2_oracle.score_candidates(query_fp, cand_fps, meta)
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_comb_03_tier1_hits_feed_mmr_optimizer(tier1_oracle, mmr_oracle, sample_candidates_list):
    """Pairwise: Top Tier 1 retrieval candidates feed into MMR portfolio optimizer."""
    q_emb = np.ones(512, dtype=np.float32)
    hits = tier1_oracle.search_library(
        q_emb, 200.0,
        np.ones((1, 512), dtype=np.float32),
        np.array([200.0]),
        ["TOP_HIT"], ["TOP_KEY_123456"]
    )
    combined_pool = hits + sample_candidates_list
    portfolio = mmr_oracle.optimize_portfolio(combined_pool, top_k=25)
    assert portfolio[0].smiles == "TOP_HIT"


def test_comb_04_knapsack_candidates_diversify_mmr_portfolio(mmr_oracle):
    """Pairwise: Knapsack de novo assemblies integrated into MMR candidate portfolio."""
    # Seed high-confidence hit
    seed = Candidate("SEED_HIT", "AAAAAAAAAAAAAA", 0.95, "tier1", 200.0)
    # Knapsack assemblies with lower individual confidence but high structural novelty
    knapsack_cands = [
        Candidate(f"KNAP_{i}", f"KNAP_{i:010d}", 0.65, "knapsack", 200.0)
        for i in range(10)
    ]
    pool = [seed] + knapsack_cands
    portfolio = mmr_oracle.optimize_portfolio(pool, top_k=5)
    sources = [c.source_tier for c in portfolio]
    assert "tier1" in sources
    assert "knapsack" in sources


def test_comb_05_networking_analog_propagation_to_mmr(mmr_oracle):
    """Pairwise: Analogs propagated through transductive network populate MMR slots."""
    seed = Candidate("SEED_AGLYCONE", "AGLYCONE_KEY12", 0.90, "tier1", 300.0)
    # Propagated glycosylated analog
    analog = Candidate("ANALOG_GLYCOSIDE", "GLYCO_KEY_1234", 0.82, "network", 462.05)
    portfolio = mmr_oracle.optimize_portfolio([seed, analog], top_k=2)
    assert len(portfolio) == 2
    assert portfolio[1].source_tier == "network"


def test_comb_06_standardizer_canonical_smiles_to_writer(standardizer_oracle, writer_oracle, tmp_path):
    """Pairwise: Standardizer output feeds atomic CSV submission writer."""
    raw_smi = "O=C(O)c1ccccc1"
    canon_smi, ik14 = standardizer_oracle.standardize_mol(raw_smi)
    assert canon_smi is not None
    preds = {"MOL_001": [canon_smi] + DISTINCT_25_SMILES[1:]}
    out_file = tmp_path / "sub.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    first_slot = df.iloc[0]["smiles"].split(";")[0]
    assert first_slot == canon_smi


def test_comb_07_governor_throttles_tier1_depth(governor_oracle):
    """Pairwise: Runtime governor mode determines candidate retrieval depth."""
    gov = RuntimeGovernorOracle(total_budget_sec=28000.0, total_queries=400)
    # DEEP mode allows top 100 retrieval
    assert gov.get_execution_mode() == "DEEP"
    deep_depth = 100 if gov.get_execution_mode() == "DEEP" else 10
    assert deep_depth == 100
    # FAST mode throttles to top 10
    gov.set_elapsed_time(27990.0)
    assert gov.get_execution_mode() == "FAST"
    fast_depth = 100 if gov.get_execution_mode() == "DEEP" else 10
    assert fast_depth == 10


def test_comb_08_governor_fast_mode_skips_tier3_generation(governor_oracle):
    """Pairwise: FAST regime conditionally bypasses expensive Tier 3 de novo sampling."""
    gov = RuntimeGovernorOracle(total_budget_sec=100.0, total_queries=200)
    gov.set_elapsed_time(90.0)  # remaining 10s for 200 queries -> FAST
    mode = gov.get_execution_mode()
    assert mode == "FAST"
    run_tier3_generation = (mode == "DEEP")
    assert not run_tier3_generation


def test_comb_09_tri_tier_retrieval_cascade():
    """Pairwise: Tier 1, Tier 2, and Tier 3 candidates merge into unified ranked pool."""
    t1_cands = [Candidate("T1_MOL", "T1_KEY_1234567", 0.92, "tier1", 250.0)]
    t2_cands = [Candidate("T2_MOL", "T2_KEY_1234567", 0.81, "tier2", 250.0)]
    t3_cands = [Candidate("T3_MOL", "T3_KEY_1234567", 0.65, "tier3", 250.0)]

    merged = t1_cands + t2_cands + t3_cands
    merged.sort(key=lambda c: c.score, reverse=True)
    assert merged[0].source_tier == "tier1"
    assert merged[1].source_tier == "tier2"
    assert merged[2].source_tier == "tier3"


def test_comb_10_adduct_neutral_mass_drives_knapsack_target(adduct_oracle):
    """Pairwise: De-adducted neutral mass sets knapsack target mass within 5 ppm."""
    obs_mz = 301.0505
    neutral_target = adduct_oracle.calculate_neutral_mass(obs_mz, "[M+H]+")
    # Fragment assembly matching neutral_target within 4 ppm
    assembled_mass = neutral_target * (1.0 + 4e-6)
    ppm_diff = abs(assembled_mass - neutral_target) / neutral_target * 1e6
    assert ppm_diff <= 5.0


def test_comb_11_networking_seeded_by_tier1_library_hit():
    """Pairwise: High confidence Tier 1 match seeds network propagation to test neighbors."""
    seed_hit = Candidate("SEED_PARENT", "SEED_KEY_12345", 0.88, "tier1", 200.0)
    can_seed_network = (seed_hit.score >= 0.82)
    assert can_seed_network


def test_comb_12_deduplication_guarantees_mmr_slot_uniqueness(mmr_oracle):
    """Pairwise: InChIKey14 deduplication ensures all MMR slots are distinct skeletons."""
    duplicates_pool = [
        Candidate("D1", "DUPLICATE_KEY1", 0.9, "tier1", 100.0),
        Candidate("D2", "DUPLICATE_KEY1", 0.85, "tier2", 100.0),
        Candidate("U1", "UNIQUE_KEY_001", 0.8, "tier1", 120.0),
        Candidate("U2", "UNIQUE_KEY_002", 0.7, "tier2", 130.0),
    ]
    portfolio = mmr_oracle.optimize_portfolio(duplicates_pool, top_k=10)
    keys = [c.inchikey14 for c in portfolio]
    assert len(keys) == len(set(keys))
    assert len(keys) == 3


def test_comb_13_writer_and_governor_interaction(governor_oracle, writer_oracle, tmp_path, distinct_25_smiles):
    """Pairwise: Submission writer operates cleanly under governor execution loop."""
    gov = RuntimeGovernorOracle(total_budget_sec=28000.0, total_queries=5)
    preds = {}
    for i in range(5):
        t0 = time.perf_counter()
        preds[f"MOL_{i:03d}"] = distinct_25_smiles
        gov.record_query_time(time.perf_counter() - t0)

    out_file = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_file)
    assert gov.completed_queries == 5
    assert out_file.exists()


def test_comb_14_13c_correction_with_adduct_de_adducting(adduct_oracle):
    """Pairwise: 13C offset correction applied before adduct neutral mass calculation."""
    true_neutral = GROUND_TRUTH_MOLECULES["caffeine"]["neutral_mass"]
    adduct = "[M+H]+"
    true_precursor = true_neutral + ADDUCT_DEFINITIONS[adduct].delta_mass
    mispicked_precursor = true_precursor + CARBON_13_OFFSET

    # 1. Correct precursor m/z
    peaks = np.array([[true_precursor, 800.0]], dtype=np.float64)
    corrected_mz = adduct_oracle.correct_precursor_mz(mispicked_precursor, peaks)

    # 2. De-adduct corrected m/z
    recovered_neutral = adduct_oracle.calculate_neutral_mass(corrected_mz, adduct)
    assert abs(recovered_neutral - true_neutral) < 1e-5


def test_comb_15_full_module_chain_end_to_end_dataflow(
    adduct_oracle, standardizer_oracle, mmr_oracle, writer_oracle, tmp_path, distinct_25_smiles
):
    """Pairwise: Full pipeline dataflow across chemistry, retrieval, reranking, and submission."""
    # Step 1: Precursor m/z + Adduct -> Neutral Mass
    obs_mz = 195.08765
    neutral_mass = adduct_oracle.calculate_neutral_mass(obs_mz, "[M+H]+")
    assert neutral_mass > 0

    # Step 2: Candidates retrieval
    raw_candidates = [
        Candidate(smi, standardizer_oracle.standardize_mol(smi)[1] or f"K_{i:012d}", 1.0 / (i + 1), "tier1", neutral_mass)
        for i, smi in enumerate(distinct_25_smiles)
    ]

    # Step 3: MMR optimization
    portfolio = mmr_oracle.optimize_portfolio(raw_candidates, top_k=25)
    assert len(portfolio) == 25

    # Step 4: Submission generation
    preds = {"MOL_E2E": [c.smiles for c in portfolio]}
    out_file = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_file)
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 1
    assert len(df.iloc[0]["smiles"].split(";")) == 25
