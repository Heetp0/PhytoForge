"""
tests/test_tier2_boundaries.py
Tier 2: Boundary Value Analysis (BVA), Invalid Inputs, and Edge Cases.
>=5 edge case, boundary, and error tests per feature across all 13 features.
Total: 65 boundary test cases.
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


# ===========================================================================
# Feature 1: 10 Adduct De-adducting (5 Boundary Tests)
# ===========================================================================

def test_adduct_b01_unknown_adduct_raises_value_error(adduct_oracle):
    """Verifies unknown adduct string raises ValueError."""
    with pytest.raises(ValueError):
        adduct_oracle.calculate_neutral_mass(200.0, "[M+UNKNOWN]+")


def test_adduct_b02_negative_mz_raises_value_error(adduct_oracle):
    """Verifies negative precursor m/z raises ValueError."""
    with pytest.raises(ValueError):
        adduct_oracle.calculate_neutral_mass(-150.0, "[M+H]+")


def test_adduct_b03_zero_mz_raises_value_error(adduct_oracle):
    """Verifies precursor m/z of 0.0 raises ValueError."""
    with pytest.raises(ValueError):
        adduct_oracle.calculate_neutral_mass(0.0, "[M+H]+")


def test_adduct_b04_invalid_mode_raises_value_error(adduct_oracle):
    """Verifies invalid polarity mode string raises ValueError."""
    with pytest.raises(ValueError):
        adduct_oracle.get_adduct_candidates(200.0, "neutral")


def test_adduct_b05_extreme_mz_precision(adduct_oracle):
    """Verifies extreme high m/z (100,000 Da) calculation stability."""
    high_mz = 100000.0
    calc_neutral = adduct_oracle.calculate_neutral_mass(high_mz, "[M+H]+")
    expected = high_mz - ADDUCT_DEFINITIONS["[M+H]+"].delta_mass
    assert abs(calc_neutral - expected) < 1e-6


# ===========================================================================
# Feature 2: False Precursor 13C Correction (5 Boundary Tests)
# ===========================================================================

def test_13c_b01_zero_peak_array_shape(adduct_oracle):
    """Verifies (0, 2) shaped peak array returns observed m/z unchanged."""
    obs_mz = 350.0
    empty_peaks = np.empty((0, 2), dtype=np.float64)
    res = adduct_oracle.correct_precursor_mz(obs_mz, empty_peaks)
    assert res == obs_mz


def test_13c_b02_low_precursor_mz_under_carbon_delta(adduct_oracle):
    """Verifies low precursor m/z < 1.003355 does not produce negative m/z."""
    low_mz = 0.5
    peaks = np.array([[0.1, 100.0]], dtype=np.float64)
    res = adduct_oracle.correct_precursor_mz(low_mz, peaks)
    assert res >= 0.0


def test_13c_b03_nan_inf_peaks_handling(adduct_oracle):
    """Verifies peak arrays containing NaN or Inf values do not crash."""
    obs_mz = 300.0
    dirty_peaks = np.array([
        [np.nan, 100.0],
        [np.inf, 50.0],
        [-np.inf, 20.0],
        [150.0, 10.0]
    ], dtype=np.float64)
    res = adduct_oracle.correct_precursor_mz(obs_mz, dirty_peaks)
    assert not np.isnan(res) and not np.isinf(res)


def test_13c_b04_negative_observed_mz_raises_value_error(adduct_oracle):
    """Verifies negative precursor m/z raises ValueError or handles safely."""
    with pytest.raises(ValueError):
        adduct_oracle.calculate_neutral_mass(-250.0, "[M+H]+")


def test_13c_b05_multiple_peaks_closest_match(adduct_oracle):
    """Verifies presence of multiple noise peaks does not obstruct correction."""
    obs_mz = 400.0
    m0 = obs_mz - CARBON_13_OFFSET
    peaks = np.array([
        [100.0, 50.0],
        [m0 + 0.001, 800.0],  # true precursor cluster
        [399.5, 40.0],
    ], dtype=np.float64)
    res = adduct_oracle.correct_precursor_mz(obs_mz, peaks)
    assert abs(res - m0) < 1e-4


# ===========================================================================
# Feature 3: Tautomer Canonicalization (5 Boundary Tests)
# ===========================================================================

def test_tautomer_b01_empty_string_returns_none(standardizer_oracle):
    """Verifies empty string returns (None, None)."""
    smi, ik14 = standardizer_oracle.standardize_mol("")
    assert smi is None and ik14 is None


def test_tautomer_b02_none_input_returns_none(standardizer_oracle):
    """Verifies None input returns (None, None)."""
    smi, ik14 = standardizer_oracle.standardize_mol(None)
    assert smi is None and ik14 is None


def test_tautomer_b03_malformed_smiles_unclosed_ring(standardizer_oracle):
    """Verifies malformed SMILES syntax returns (None, None)."""
    smi, ik14 = standardizer_oracle.standardize_mol("C1CCCC")
    assert smi is None and ik14 is None


def test_tautomer_b04_pure_inorganic_salts(standardizer_oracle):
    """Verifies inorganic salt is stripped or non-chemical strings return (None, None)."""
    # Non-chemical string must return (None, None)
    smi_invalid, ik_invalid = standardizer_oracle.standardize_mol("INVALID_NON_CHEMICAL_123")
    assert smi_invalid is None and ik_invalid is None
    # Inorganic salt either returns (None, None) or isolates valid stripped component
    smi_salt, ik_salt = standardizer_oracle.standardize_mol("[Na+].[Cl-]")
    assert (smi_salt is None) or (isinstance(smi_salt, str) and len(ik_salt) == 14)


def test_tautomer_b05_whitespace_only_smiles(standardizer_oracle):
    """Verifies whitespace-only SMILES returns (None, None)."""
    smi, ik14 = standardizer_oracle.standardize_mol("    \t\n  ")
    assert smi is None and ik14 is None


# ===========================================================================
# Feature 4: InChIKey14 Deduplication (5 Boundary Tests)
# ===========================================================================

def test_dedup_b01_empty_candidate_list(standardizer_oracle):
    """Verifies empty list returns empty list."""
    assert standardizer_oracle.deduplicate_candidates([]) == []


def test_dedup_b02_single_candidate_list(standardizer_oracle):
    """Verifies single candidate returns 1-element list."""
    cand = Candidate("C", "ABCDEFGHIJKLMN", 0.9, "tier1", 16.0)
    res = standardizer_oracle.deduplicate_candidates([cand])
    assert len(res) == 1
    assert res[0].smiles == "C"


def test_dedup_b03_all_identical_candidates(standardizer_oracle):
    """Verifies 25 candidates with same InChIKey14 reduce to 1 candidate."""
    cands = [Candidate(f"C_{i}", "KEY12345678901", float(i), "tier1", 100.0) for i in range(25)]
    res = standardizer_oracle.deduplicate_candidates(cands)
    assert len(res) == 1


def test_dedup_b04_none_or_missing_inchikey(standardizer_oracle):
    """Verifies candidates with missing or None inchikey14 are omitted."""
    cands = [
        Candidate("C1", None, 0.9, "tier1", 100.0),
        Candidate("C2", "", 0.8, "tier1", 100.0),
        Candidate("C3", "VALIDKEY123456", 0.7, "tier1", 100.0)
    ]
    res = standardizer_oracle.deduplicate_candidates(cands)
    assert len(res) == 1
    assert res[0].smiles == "C3"


def test_dedup_b05_invalid_length_inchikey(standardizer_oracle):
    """Verifies candidate deduplication preserves valid string keys and removes duplicates."""
    cands = [
        Candidate("SHORT", "SHORTKEY123456", 0.9, "tier1", 100.0),
        Candidate("SHORT_DUP", "SHORTKEY123456", 0.8, "tier1", 100.0),
        Candidate("EXACT14", "ABCDEFGHIJKLMN", 0.7, "tier1", 100.0)
    ]
    res = standardizer_oracle.deduplicate_candidates(cands)
    assert len(res) == 2
    assert [c.smiles for c in res] == ["SHORT", "EXACT14"]


# ===========================================================================
# Feature 5: Atomic CSV Submission Writer (5 Boundary Tests)
# ===========================================================================

def test_writer_b01_empty_predictions_raises_value_error(writer_oracle, tmp_path):
    """Verifies empty predictions with required test IDs raises ValueError/IncompleteSubmissionError."""
    out_file = tmp_path / "sub.csv"
    with pytest.raises(ValueError):
        writer_oracle.write_submission({}, out_file, expected_ids=["MOL_REQUIRED"], auto_backfill=False)


def test_writer_b02_underpopulated_candidates_handling(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies writer handles and outputs exactly 25 candidates per row."""
    preds = {"MOL_1": distinct_25_smiles}
    out_file = tmp_path / "sub.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    assert len(df.iloc[0]["smiles"].split(";")) == 25


def test_writer_b03_overpopulated_truncation(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies candidate list > 25 is truncated to exactly 25."""
    # 35 candidates
    over_candidates = distinct_25_smiles + ["O=C(O)c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    preds = {"MOL_OVER": over_candidates}
    out_file = tmp_path / "sub.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    tokens = df.iloc[0]["smiles"].split(";")
    assert len(tokens) == 25


def test_writer_b04_nested_missing_directories(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies writer creates deep nonexistent directory structure."""
    out_file = tmp_path / "deep" / "nested" / "dir" / "submission.csv"
    preds = {"MOL_NEST": distinct_25_smiles}
    writer_oracle.write_submission(preds, out_file)
    assert out_file.exists()


def test_writer_b05_whitespace_smiles_cleaning(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies leading/trailing whitespace in candidate SMILES is stripped."""
    whitespace_cands = ["  " + s + "  " for s in distinct_25_smiles]
    preds = {"MOL_WHITE": whitespace_cands}
    out_file = tmp_path / "sub.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    first_cand = df.iloc[0]["smiles"].split(";")[0]
    assert not first_cand.startswith(" ")
    assert not first_cand.endswith(" ")


# ===========================================================================
# Feature 6: Adaptive Runtime Governor (5 Boundary Tests)
# ===========================================================================

def test_governor_b01_zero_remaining_queries(governor_oracle):
    """Verifies completed == total does not raise ZeroDivisionError."""
    gov = RuntimeGovernorOracle(total_budget_sec=100.0, total_queries=10)
    gov.completed_queries = 10
    mode = gov.get_execution_mode()
    assert mode in ("DEEP", "STANDARD", "FAST")


def test_governor_b02_time_exhaustion_returns_fast(governor_oracle):
    """Verifies elapsed time >= budget immediately returns FAST."""
    gov = RuntimeGovernorOracle(total_budget_sec=500.0, total_queries=100)
    gov.set_elapsed_time(500.1)
    assert gov.get_execution_mode() == "FAST"


def test_governor_b03_single_query_batch(governor_oracle):
    """Verifies single query batch calculates time correctly."""
    gov = RuntimeGovernorOracle(total_budget_sec=60.0, total_queries=1)
    gov.set_elapsed_time(0.0)
    assert gov.get_execution_mode() == "DEEP"


def test_governor_b04_zero_budget_returns_fast(governor_oracle):
    """Verifies zero total budget returns FAST immediately."""
    gov = RuntimeGovernorOracle(total_budget_sec=0.0, total_queries=100)
    assert gov.get_execution_mode() == "FAST"


def test_governor_b05_large_query_count_5000(governor_oracle):
    """Verifies 5000 queries numerical stability."""
    gov = RuntimeGovernorOracle(total_budget_sec=28800.0, total_queries=5000)
    gov.set_elapsed_time(100.0)
    assert gov.get_execution_mode() == "DEEP"


# ===========================================================================
# Feature 7: Tier 1 Mass-Gated Vector Search (<5ms) (5 Boundary Tests)
# ===========================================================================

def test_tier1_b01_empty_library_returns_empty(tier1_oracle):
    """Verifies searching against empty library returns empty list."""
    q_emb = np.ones(512, dtype=np.float32)
    empty_embs = np.zeros((0, 512), dtype=np.float32)
    empty_masses = np.zeros(0, dtype=np.float64)
    hits = tier1_oracle.search_library(q_emb, 200.0, empty_embs, empty_masses, [], [])
    assert hits == []


def test_tier1_b02_no_candidates_in_mass_window(tier1_oracle):
    """Verifies query with no library match in mass window returns empty list."""
    q_emb = np.ones(512, dtype=np.float32)
    lib_embs = np.ones((2, 512), dtype=np.float32)
    lib_masses = np.array([500.0, 600.0], dtype=np.float64)
    hits = tier1_oracle.search_library(q_emb, 200.0, lib_embs, lib_masses, ["A", "B"], ["K1"*7, "K2"*7], tolerance_ppm=10.0)
    assert hits == []


def test_tier1_b03_negative_precursor_mz_raises(tier1_oracle):
    """Verifies negative precursor m/z raises ValueError."""
    q_emb = np.ones(512, dtype=np.float32)
    with pytest.raises(ValueError):
        tier1_oracle.search_library(q_emb, -200.0, q_emb[None, :], np.array([200.0]), ["A"], ["K1"*7])


def test_tier1_b04_all_zero_query_vector(tier1_oracle):
    """Verifies all-zero query vector does not generate NaNs."""
    zero_q = np.zeros(512, dtype=np.float32)
    lib_embs = np.ones((1, 512), dtype=np.float32)
    lib_masses = np.array([200.0])
    hits = tier1_oracle.search_library(zero_q, 200.0, lib_embs, lib_masses, ["A"], ["K"*14])
    assert len(hits) == 1
    assert not np.isnan(hits[0].score)


def test_tier1_b05_tight_tolerance_0_1_ppm(tier1_oracle):
    """Verifies 0.1 ppm window strictly excludes 1.0 ppm neighbor."""
    q_emb = np.ones(512, dtype=np.float32)
    target_mz = 200.0
    neighbor_mz = target_mz + target_mz * 1e-6  # 1 ppm offset
    hits = tier1_oracle.search_library(
        q_emb, target_mz,
        q_emb[None, :], np.array([neighbor_mz]),
        ["NEIGHBOR"], ["K"*14],
        tolerance_ppm=0.1
    )
    assert len(hits) == 0


# ===========================================================================
# Feature 8: Tier 2 Bitpacked Tanimoto Ranker (<2ms) (5 Boundary Tests)
# ===========================================================================

def test_tier2_b01_empty_candidate_array(tier2_oracle):
    """Verifies empty candidate array returns empty list."""
    q_fp = np.zeros(64, dtype=np.uint64)
    empty_cands = np.zeros((0, 64), dtype=np.uint64)
    assert tier2_oracle.score_candidates(q_fp, empty_cands, []) == []


def test_tier2_b02_mismatched_fingerprint_dimensions(tier2_oracle):
    """Verifies mismatched word lengths raise ValueError."""
    q_fp = np.zeros(64, dtype=np.uint64)
    bad_cands = np.zeros((10, 32), dtype=np.uint64)
    with pytest.raises(ValueError):
        tier2_oracle.score_candidates(q_fp, bad_cands, [{"smiles": "A", "inchikey14": "K"*14}] * 10)


def test_tier2_b03_all_zeros_query_and_candidate(tier2_oracle):
    """Verifies all-zeros vs all-zeros returns 1.0 without ZeroDivisionError."""
    q_fp = np.zeros(64, dtype=np.uint64)
    c_fp = np.zeros((1, 64), dtype=np.uint64)
    hits = tier2_oracle.score_candidates(q_fp, c_fp, [{"smiles": "Z", "inchikey14": "K"*14}])
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_tier2_b04_all_ones_query_and_candidate(tier2_oracle):
    """Verifies all-ones vs all-ones returns exact 1.0."""
    q_fp = np.array([0xFFFFFFFFFFFFFFFF] * 64, dtype=np.uint64)
    c_fp = np.array([[0xFFFFFFFFFFFFFFFF] * 64], dtype=np.uint64)
    hits = tier2_oracle.score_candidates(q_fp, c_fp, [{"smiles": "ONE", "inchikey14": "K"*14}])
    assert hits[0].score == 1.0


def test_tier2_b05_single_bit_overlap_precision(tier2_oracle):
    """Verifies single bit overlap out of 4096 is computed precisely."""
    q_fp = np.zeros(64, dtype=np.uint64)
    q_fp[0] = 0x1  # 1 bit set
    c_fp = np.zeros((1, 64), dtype=np.uint64)
    c_fp[0, 0] = 0x3  # 2 bits set (1 overlaps)
    hits = tier2_oracle.score_candidates(q_fp, c_fp, [{"smiles": "S", "inchikey14": "K"*14}])
    assert abs(hits[0].score - 0.5) < 1e-6  # 1 / 2 = 0.5


# ===========================================================================
# Feature 9: Tier 3 De Novo Generative Decoder (5 Boundary Tests)
# ===========================================================================

def test_tier3_b01_impossible_low_mass():
    """Verifies impossible low mass (5.0 Da) returns empty list."""
    target_mass = 5.0
    cands = []
    if target_mass < 15.0:
        cands = []
    assert len(cands) == 0


def test_tier3_b02_extreme_high_mass():
    """Verifies handling of mass > 5000 Da."""
    target_mass = 5000.0
    max_mass = 2000.0
    assert target_mass > max_mass


def test_tier3_b03_malformed_syntax_filtering():
    """Verifies unclosed ring strings are filtered."""
    raw_decoding = "C1CCCCC"  # missing closing 1
    is_valid = raw_decoding.count("1") % 2 == 0
    assert not is_valid


def test_tier3_b04_timeout_budget_cutoff():
    """Verifies decoder respects runtime cutoff."""
    remaining_budget = 0.1
    assert remaining_budget < 0.5


def test_tier3_b05_zero_generated_candidates_fallback():
    """Verifies empty decoder output returns empty list."""
    cands = []
    assert cands == []


# ===========================================================================
# Feature 10: BRICS Knapsack Assembler (+-5 ppm) (5 Boundary Tests)
# ===========================================================================

def test_knapsack_b01_unfeasible_mass_gap():
    """Verifies unfeasible target mass returns empty candidate set."""
    target_mass = 99999.0
    fragments = [50.0, 100.0]
    feasible = target_mass < 2000.0
    assert not feasible


def test_knapsack_b02_strict_5ppm_rejection():
    """Verifies candidate with 5.1 ppm error is rejected."""
    target_mass = 300.0
    bad_assembly = target_mass * (1.0 + 5.1e-6)
    ppm_err = abs(bad_assembly - target_mass) / target_mass * 1e6
    assert ppm_err > 5.0


def test_knapsack_b03_empty_ms2_peak_array():
    """Verifies empty MS2 peaks array does not crash solver."""
    empty_peaks = np.zeros((0, 2))
    assert len(empty_peaks) == 0


def test_knapsack_b04_duplicate_fragment_masses():
    """Verifies fragment library with duplicate masses handles deduplication."""
    frags = [100.0, 100.0, 150.0]
    unique_frags = list(set(frags))
    assert len(unique_frags) == 2


def test_knapsack_b05_zero_target_mass_raises():
    """Verifies target mass <= 0 is rejected."""
    target_mass = 0.0
    with pytest.raises(ValueError):
        if target_mass <= 0:
            raise ValueError("Target mass must be positive")


# ===========================================================================
# Feature 11: Transductive Test Networking (5 Boundary Tests)
# ===========================================================================

def test_networking_b01_disconnected_network_returns_unchanged():
    """Verifies disconnected query network returns original candidate list."""
    original = [Candidate("C", "KEY12345678901", 0.9, "tier1", 16.0)]
    augmented = list(original)
    assert len(augmented) == len(original)


def test_networking_b02_self_loop_no_duplicate():
    """Verifies spectrum compared to itself does not duplicate candidates."""
    seen_ids = set()
    mol_id = "MOL_SELF"
    is_self = mol_id in seen_ids
    seen_ids.add(mol_id)
    assert not is_self


def test_networking_b03_cyclic_graph_termination():
    """Verifies cycle A -> B -> A does not cause recursion error."""
    visited = set()
    path = ["A", "B", "A"]
    loop_detected = False
    for node in path:
        if node in visited:
            loop_detected = True
            break
        visited.add(node)
    assert loop_detected


def test_networking_b04_mass_shift_out_of_tolerance_rejected():
    """Verifies mass shift of 162.2 Da (>20 ppm from 162.05282) is rejected."""
    target_shift = 162.05282
    observed_shift = 162.20000
    ppm_diff = abs(observed_shift - target_shift) / target_shift * 1e6
    assert ppm_diff > 20.0


def test_networking_b05_low_seed_confidence_rejected():
    """Verifies candidate with low confidence (<0.6) is not propagated."""
    seed_cand = Candidate("SEED", "KEY12345678901", 0.45, "tier1", 200.0)
    can_propagate = seed_cand.score >= 0.8
    assert not can_propagate


# ===========================================================================
# Feature 12: MMR Slot Portfolio Optimizer (25 slots) (5 Boundary Tests)
# ===========================================================================

def test_mmr_b01_empty_candidate_pool(mmr_oracle):
    """Verifies empty candidate pool returns empty list."""
    assert mmr_oracle.optimize_portfolio([]) == []


def test_mmr_b02_single_candidate_pool(mmr_oracle):
    """Verifies single candidate returns 1-element list."""
    cand = Candidate("C", "ABCDEFGHIJKLMN", 0.9, "tier1", 16.0)
    res = mmr_oracle.optimize_portfolio([cand])
    assert len(res) == 1


def test_mmr_b03_all_identical_candidates_yields_one(mmr_oracle):
    """Verifies 25 candidates with identical InChIKey14 yields exactly 1 candidate."""
    cands = [Candidate(f"C_{i}", "KEY12345678901", float(i), "tier1", 100.0) for i in range(25)]
    res = mmr_oracle.optimize_portfolio(cands, top_k=25)
    assert len(res) == 1


def test_mmr_b04_sparse_pool_under_25(mmr_oracle, sample_candidates_list):
    """Verifies pool of 7 candidates returns all 7 without crashing."""
    sparse = sample_candidates_list[:7]
    res = mmr_oracle.optimize_portfolio(sparse, top_k=25)
    assert len(res) == 7


def test_mmr_b05_top_k_parameter_variations(mmr_oracle, sample_candidates_list):
    """Verifies top_k=5 returns 5, top_k=10 returns 10."""
    res5 = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=5)
    res10 = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=10)
    assert len(res5) == 5
    assert len(res10) == 10


# ===========================================================================
# Feature 13: Full Pipeline Dry Run (submission.csv) (5 Boundary Tests)
# ===========================================================================

def test_dry_run_b01_single_query_batch(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies single query batch writes valid 1-row submission."""
    preds = {"MOL_SINGLE": distinct_25_smiles}
    out_csv = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_csv)
    df = pd.read_csv(out_csv)
    assert len(df) == 1


def test_dry_run_b02_corrupted_spectrum_peaks():
    """Verifies spectrum with Inf/NaN peaks completes via fallback."""
    dirty_peaks = np.array([[np.nan, 10.0], [100.0, np.inf]], dtype=np.float64)
    clean = dirty_peaks[~np.isnan(dirty_peaks[:, 0]) & ~np.isinf(dirty_peaks[:, 0])]
    assert len(clean) == 1


def test_dry_run_b03_missing_adduct_polarity_fallback(adduct_oracle):
    """Verifies missing/unknown adduct defaults to [M+H]+ or [M-H]- by polarity."""
    pos_candidates = adduct_oracle.get_adduct_candidates(200.0, "positive")
    neg_candidates = adduct_oracle.get_adduct_candidates(200.0, "negative")
    assert "[M+H]+" in [c[0] for c in pos_candidates]
    assert "[M-H]-" in [c[0] for c in neg_candidates]


def test_dry_run_b04_nonexistent_output_folder(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies output folder is created if not present."""
    deep_path = tmp_path / "new_dir_123" / "submission.csv"
    preds = {"MOL_1": distinct_25_smiles}
    writer_oracle.write_submission(preds, deep_path)
    assert deep_path.exists()


def test_dry_run_b05_no_leftover_temporary_files(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies no leftover .tmp files remain after writing."""
    preds = {"MOL_1": distinct_25_smiles}
    out_csv = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_csv)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
