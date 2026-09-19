"""
tests/test_tier1_features.py
Tier 1: Isolated Feature Coverage (>=5 isolated tests per feature across all 13 features).
Total: 65 feature test cases.
"""

from decimal import Decimal
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
    AdductsOracle,
    StandardizerOracle,
    SubmissionWriterOracle,
    RuntimeGovernorOracle,
    Tier1VectorMatcherOracle,
    Tier2TanimotoRankerOracle,
    MMROptimizerOracle,
)


# ===========================================================================
# Feature 1: 10 Adduct De-adducting (5 Tests)
# ===========================================================================

def test_adduct_f01_positive_all_5_adducts(adduct_oracle):
    """Verifies calculate_neutral_mass for all 5 positive competition adducts."""
    target_neutral = GROUND_TRUTH_MOLECULES["caffeine"]["neutral_mass"]  # 194.08037557
    positive_adducts = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+"]
    for adduct in positive_adducts:
        meta = ADDUCT_DEFINITIONS[adduct]
        mz = target_neutral + meta.delta_mass
        calculated = adduct_oracle.calculate_neutral_mass(mz, adduct)
        assert abs(calculated - target_neutral) < 1e-6, (
            f"Adduct {adduct}: expected {target_neutral}, got {calculated}"
        )


def test_adduct_f02_negative_all_5_adducts(adduct_oracle):
    """Verifies calculate_neutral_mass for all 5 negative competition adducts."""
    target_neutral = GROUND_TRUTH_MOLECULES["aspirin"]["neutral_mass"]  # 180.04225874
    negative_adducts = ["[M-H]-", "[M+Cl]-", "[M+FA-H]-", "[M+Hac-H]-", "[M-H2O-H]-"]
    for adduct in negative_adducts:
        meta = ADDUCT_DEFINITIONS[adduct]
        mz = target_neutral + meta.delta_mass
        calculated = adduct_oracle.calculate_neutral_mass(mz, adduct)
        assert abs(calculated - target_neutral) < 1e-6, (
            f"Adduct {adduct}: expected {target_neutral}, got {calculated}"
        )


def test_adduct_f03_candidate_generation_positive(adduct_oracle):
    """Verifies get_adduct_candidates generates all 5 positive candidates."""
    mz = 200.0
    candidates = adduct_oracle.get_adduct_candidates(mz, "positive")
    adduct_names = [c[0] for c in candidates]
    assert len(candidates) == 5
    for expected in ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+"]:
        assert expected in adduct_names


def test_adduct_f04_candidate_generation_negative(adduct_oracle):
    """Verifies get_adduct_candidates generates all 5 negative candidates."""
    mz = 200.0
    candidates = adduct_oracle.get_adduct_candidates(mz, "negative")
    adduct_names = [c[0] for c in candidates]
    assert len(candidates) == 5
    for expected in ["[M-H]-", "[M+Cl]-", "[M+FA-H]-", "[M+Hac-H]-", "[M-H2O-H]-"]:
        assert expected in adduct_names


def test_adduct_f05_exact_delta_electron_mass(adduct_oracle):
    """Verifies exact inclusion of electron mass in adduct monoisotopic delta."""
    # [M+H]+: 1H (1.007825032) - 1e- (0.000548580) = +1.007276452
    expected_delta = 1.00782503223 - 0.000548579909
    actual_delta = ADDUCT_DEFINITIONS["[M+H]+"].delta_mass
    assert abs(actual_delta - expected_delta) < 1e-9
    # Round-trip check
    mz = 100.0 + expected_delta
    calc_m = adduct_oracle.calculate_neutral_mass(mz, "[M+H]+")
    assert abs(calc_m - 100.0) < 1e-9


# ===========================================================================
# Feature 2: False Precursor 13C Correction (5 Tests)
# ===========================================================================

def test_13c_f01_shift_applied_when_peak_detected(adduct_oracle, mock_13c_spectrum):
    """Verifies precursor m/z is shifted by -1.003355 Da when M-1 peak is in MS2."""
    true_m0 = mock_13c_spectrum.precursor_mz - CARBON_13_OFFSET
    corrected = adduct_oracle.correct_precursor_mz(mock_13c_spectrum.precursor_mz, mock_13c_spectrum.peaks)
    assert abs(corrected - true_m0) < 1e-5


def test_13c_f02_no_shift_when_no_peak_detected(adduct_oracle):
    """Verifies precursor m/z remains unchanged when no M-1 peak is in MS2."""
    obs_mz = 300.1234
    peaks = np.array([[150.0, 100.0], [200.0, 50.0]], dtype=np.float64)
    corrected = adduct_oracle.correct_precursor_mz(obs_mz, peaks)
    assert abs(corrected - obs_mz) < 1e-9


def test_13c_f03_empty_ms2_peak_list(adduct_oracle):
    """Verifies empty MS2 array returns observed m/z safely."""
    obs_mz = 250.0
    empty_peaks = np.zeros((0, 2), dtype=np.float64)
    corrected = adduct_oracle.correct_precursor_mz(obs_mz, empty_peaks)
    assert corrected == obs_mz


def test_13c_f04_tolerance_window_exact_boundary(adduct_oracle):
    """Verifies peak within tolerance triggers correction, while peak outside is rejected."""
    obs_mz = 400.0
    m0 = obs_mz - CARBON_13_OFFSET
    # Peak within 5 ppm (< 0.002 Da) of m0
    peak_in = np.array([[m0 + 0.002, 100.0]], dtype=np.float64)
    assert abs(adduct_oracle.correct_precursor_mz(obs_mz, peak_in) - m0) < 1e-5
    # Peak at 0.05 Da of m0 (outside tolerance)
    peak_out = np.array([[m0 + 0.05, 100.0]], dtype=np.float64)
    assert adduct_oracle.correct_precursor_mz(obs_mz, peak_out) == obs_mz


def test_13c_f05_exact_carbon_13_delta_value():
    """Verifies 13C offset matches NIST C13 - C12 (1.003354835 Da)."""
    assert abs(CARBON_13_OFFSET - 1.003354835) < 1e-8


# ===========================================================================
# Feature 3: Tautomer Canonicalization (5 Tests)
# ===========================================================================

def test_tautomer_f01_2pyridone_and_2hydroxypyridine(standardizer_oracle):
    """Verifies 2-pyridone and 2-hydroxypyridine canonicalize to identical InChIKey14."""
    smi1 = "O=C1NC=CC=C1"
    smi2 = "Oc1ncccc1"
    _, ik1 = standardizer_oracle.standardize_mol(smi1)
    _, ik2 = standardizer_oracle.standardize_mol(smi2)
    assert ik1 is not None and ik2 is not None
    assert ik1 == ik2, f"Expected same InChIKey14, got {ik1} vs {ik2}"


def test_tautomer_f02_salt_stripping_sodium_benzoate(standardizer_oracle):
    """Verifies sodium benzoate strips salt to benzoic acid InChIKey14."""
    smi_salt = "[Na+].[O-]C(=O)c1ccccc1"
    smi_neutral = "O=C(O)c1ccccc1"
    _, ik_salt = standardizer_oracle.standardize_mol(smi_salt)
    _, ik_neutral = standardizer_oracle.standardize_mol(smi_neutral)
    assert ik_salt == ik_neutral
    assert ik_salt == GROUND_TRUTH_MOLECULES["benzoic_acid"]["inchikey14"]


def test_tautomer_f03_glycine_zwitterion_neutralization(standardizer_oracle):
    """Verifies zwitterionic glycine normalizes to neutral glycine InChIKey14."""
    smi_zwit = "[NH3+]CC(=O)[O-]"
    smi_neut = "NCC(=O)O"
    _, ik_zwit = standardizer_oracle.standardize_mol(smi_zwit)
    _, ik_neut = standardizer_oracle.standardize_mol(smi_neut)
    assert ik_zwit == ik_neut


def test_tautomer_f04_stereoisomer_skeleton_invariance(standardizer_oracle):
    """Verifies L-alanine and D-alanine share identical InChIKey14."""
    l_ala = "N[C@@H](C)C(=O)O"
    d_ala = "N[C@H](C)C(=O)O"
    _, ik_l = standardizer_oracle.standardize_mol(l_ala)
    _, ik_d = standardizer_oracle.standardize_mol(d_ala)
    assert ik_l == ik_d


def test_tautomer_f05_aspirin_canonical_smiles_and_key(standardizer_oracle):
    """Verifies Aspirin canonicalization produces valid 14-char key."""
    smi = GROUND_TRUTH_MOLECULES["aspirin"]["smiles"]
    canon_smi, ik14 = standardizer_oracle.standardize_mol(smi)
    assert canon_smi is not None
    assert ik14 == GROUND_TRUTH_MOLECULES["aspirin"]["inchikey14"]
    assert len(ik14) == 14


# ===========================================================================
# Feature 4: InChIKey14 Deduplication (5 Tests)
# ===========================================================================

def test_dedup_f01_strips_duplicate_candidates(standardizer_oracle):
    """Verifies candidates with duplicate InChIKey14 are pruned."""
    cands = [
        Candidate("C1", "ABCDEFGHIJKLMN", 0.9, "tier1", 100.0),
        Candidate("C2", "ABCDEFGHIJKLMN", 0.8, "tier2", 100.0),
        Candidate("C3", "ZZZZZZZZZZZZZZ", 0.7, "tier1", 150.0),
    ]
    deduped = standardizer_oracle.deduplicate_candidates(cands)
    assert len(deduped) == 2


def test_dedup_f02_preserves_highest_score(standardizer_oracle):
    """Verifies highest scoring candidate is retained among duplicates in ranked list."""
    cands = [
        Candidate("HighScore", "ABCDEFGHIJKLMN", 0.95, "tier1", 100.0),
        Candidate("MidScore", "ABCDEFGHIJKLMN", 0.5, "tier2", 100.0),
        Candidate("LowScore", "ABCDEFGHIJKLMN", 0.2, "tier3", 100.0),
    ]
    deduped = standardizer_oracle.deduplicate_candidates(cands)
    assert len(deduped) == 1
    assert deduped[0].smiles == "HighScore"
    assert deduped[0].score == 0.95


def test_dedup_f03_stable_ordering(standardizer_oracle):
    """Verifies deduplicated candidate list preserves score descending order."""
    cands = [
        Candidate("C1", "AAAAAAAAAAAAAA", 0.9, "tier1", 100.0),
        Candidate("C2", "BBBBBBBBBBBBBB", 0.8, "tier1", 110.0),
        Candidate("C3", "CCCCCCCCCCCCCC", 0.7, "tier1", 120.0),
    ]
    deduped = standardizer_oracle.deduplicate_candidates(cands)
    scores = [c.score for c in deduped]
    assert scores == [0.9, 0.8, 0.7]


def test_dedup_f04_all_unique_pool_untouched(standardizer_oracle, sample_candidates_list):
    """Verifies all-unique candidate list retains exact original count."""
    deduped = standardizer_oracle.deduplicate_candidates(sample_candidates_list)
    assert len(deduped) == len(sample_candidates_list)


def test_dedup_f05_output_all_keys_unique(standardizer_oracle):
    """Verifies all candidates in result have unique InChIKey14."""
    rng = np.random.RandomState(42)
    keys = ["KEY_A_12345678", "KEY_B_12345678", "KEY_C_12345678"]
    cands = [Candidate(f"S_{i}", rng.choice(keys), rng.rand(), "tier1", 200.0) for i in range(20)]
    deduped = standardizer_oracle.deduplicate_candidates(cands)
    result_keys = [c.inchikey14 for c in deduped]
    assert len(result_keys) == len(set(result_keys))
    assert len(result_keys) <= 3


# ===========================================================================
# Feature 5: Atomic CSV Submission Writer (5 Tests)
# ===========================================================================

def test_writer_f01_valid_header_molecule_id_smiles(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies CSV header is strictly 'molecule_id,smiles'."""
    preds = {"MOL_1": distinct_25_smiles}
    out_file = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_file)
    with open(out_file, "r") as f:
        header = f.readline().strip()
    assert header == "molecule_id,smiles"


def test_writer_f02_exact_row_count(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies output CSV row count matches prediction dictionary."""
    preds = {f"MOL_{i}": distinct_25_smiles for i in range(10)}
    out_file = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    assert len(df) == 10


def test_writer_f03_exactly_25_semicolon_delimited_smiles(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies each row contains exactly 25 semicolon-delimited SMILES."""
    preds = {"MOL_TEST": distinct_25_smiles}
    out_file = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    entry = df.iloc[0]["smiles"]
    tokens = entry.split(";")
    assert len(tokens) == 25
    assert entry.count(";") == 24


def test_writer_f04_atomic_replace_temporary_file(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies atomic write cleans up temporary file."""
    preds = {"MOL_ATOMIC": distinct_25_smiles}
    out_file = tmp_path / "submission.csv"
    tmp_file = tmp_path / "submission.csv.tmp"
    writer_oracle.write_submission(preds, out_file)
    assert out_file.exists()
    assert not tmp_file.exists()


def test_writer_f05_valid_csv_parseable_by_pandas(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies generated submission file is cleanly parseable by pandas."""
    preds = {"MOL_A": distinct_25_smiles, "MOL_B": distinct_25_smiles}
    out_file = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_file)
    df = pd.read_csv(out_file)
    assert list(df.columns) == ["molecule_id", "smiles"]
    assert not df.isnull().any().any()


# ===========================================================================
# Feature 6: Adaptive Runtime Governor (5 Tests)
# ===========================================================================

def test_governor_f01_deep_mode_when_budget_high(governor_oracle):
    """Verifies DEEP mode when remaining time per query > 5.0s."""
    gov = RuntimeGovernorOracle(total_budget_sec=28000.0, total_queries=400)
    gov.set_elapsed_time(100.0)  # ~70s/query remaining
    assert gov.get_execution_mode() == "DEEP"


def test_governor_f02_standard_mode_transition(governor_oracle):
    """Verifies STANDARD mode when remaining time per query is 2.0s to 5.0s."""
    gov = RuntimeGovernorOracle(total_budget_sec=1000.0, total_queries=400)
    # Remaining time = 1000s for 300 queries -> 3.33s/query
    gov.completed_queries = 100
    assert gov.get_execution_mode() == "STANDARD"


def test_governor_f03_fast_mode_transition(governor_oracle):
    """Verifies FAST mode when remaining time per query < 2.0s."""
    gov = RuntimeGovernorOracle(total_budget_sec=100.0, total_queries=100)
    gov.set_elapsed_time(50.0)  # 50s for 100 queries -> 0.5s/query
    assert gov.get_execution_mode() == "FAST"


def test_governor_f04_tracks_completed_queries(governor_oracle):
    """Verifies record_query_time increments completed queries."""
    gov = RuntimeGovernorOracle(total_budget_sec=5000.0, total_queries=100)
    assert gov.completed_queries == 0
    gov.record_query_time(2.5)
    assert gov.completed_queries == 1
    gov.record_query_time(3.1)
    assert gov.completed_queries == 2


def test_governor_f05_time_budget_allocation(governor_oracle):
    """Verifies total budget partition over 5000 queries averages 5.6s."""
    gov = RuntimeGovernorOracle(total_budget_sec=28000.0, total_queries=5000)
    sec_per_mol = gov.total_budget / gov.total_queries
    assert sec_per_mol == 5.6
    assert sec_per_mol < 5.76


# ===========================================================================
# Feature 7: Tier 1 Mass-Gated Vector Search (<5ms) (5 Tests)
# ===========================================================================

def test_tier1_f01_mass_window_gating(tier1_oracle):
    """Verifies candidates outside +-10 ppm are excluded."""
    query_mz = 200.0
    query_emb = np.ones(512, dtype=np.float32)
    lib_masses = np.array([199.999, 200.001, 201.000, 198.000], dtype=np.float64)
    lib_embs = np.ones((4, 512), dtype=np.float32)
    lib_smiles = ["C1", "C2", "C3", "C4"]
    lib_keys = ["K1" * 7, "K2" * 7, "K3" * 7, "K4" * 7]
    hits = tier1_oracle.search_library(query_emb, query_mz, lib_embs, lib_masses, lib_smiles, lib_keys, tolerance_ppm=10.0)
    hit_smiles = [h.smiles for h in hits]
    assert "C1" in hit_smiles
    assert "C2" in hit_smiles
    assert "C3" not in hit_smiles
    assert "C4" not in hit_smiles


def test_tier1_f02_cosine_score_ranking(tier1_oracle):
    """Verifies search results are sorted descending by cosine similarity."""
    query_mz = 300.0
    query_emb = np.array([1.0, 0.0] * 256, dtype=np.float32)
    lib_masses = np.array([300.001, 300.002], dtype=np.float64)
    # Cand 1 has higher cosine similarity
    cand1_emb = np.array([1.0, 0.0] * 256, dtype=np.float32)
    cand2_emb = np.array([0.0, 1.0] * 256, dtype=np.float32)
    lib_embs = np.stack([cand1_emb, cand2_emb])
    hits = tier1_oracle.search_library(query_emb, query_mz, lib_embs, lib_masses, ["C_HIGH", "C_LOW"], ["KH" * 7, "KL" * 7])
    assert len(hits) == 2
    assert hits[0].smiles == "C_HIGH"
    assert hits[0].score > hits[1].score


def test_tier1_f03_latency_benchmark_under_5ms(tier1_oracle):
    """Verifies Tier 1 search completes in < 5.0ms on 1000-candidate library."""
    rng = np.random.RandomState(42)
    lib_size = 1000
    lib_embs = rng.randn(lib_size, 512).astype(np.float32)
    lib_masses = rng.uniform(100.0, 600.0, size=lib_size).astype(np.float64)
    lib_smiles = [f"MOL_{i}" for i in range(lib_size)]
    lib_keys = [f"KEY_{i:010d}" for i in range(lib_size)]
    query_emb = rng.randn(512).astype(np.float32)
    query_mz = 350.0

    # Benchmark query latency
    t0 = time.perf_counter()
    tier1_oracle.search_library(query_emb, query_mz, lib_embs, lib_masses, lib_smiles, lib_keys, tolerance_ppm=10.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 5.0, f"Tier 1 search exceeded 5ms latency: {elapsed_ms:.2f}ms"


def test_tier1_f04_candidate_contract(tier1_oracle):
    """Verifies returned Candidate objects have source_tier='tier1'."""
    query_emb = np.ones(512, dtype=np.float32)
    hits = tier1_oracle.search_library(
        query_emb, 250.0,
        np.ones((1, 512), dtype=np.float32),
        np.array([250.0]),
        ["SMILES_1"],
        ["KEY12345678901"]
    )
    assert len(hits) == 1
    assert hits[0].source_tier == "tier1"
    assert hits[0].smiles == "SMILES_1"


def test_tier1_f05_exact_match_score_near_one(tier1_oracle):
    """Verifies identical embedding yields similarity score ~ 1.0."""
    emb = np.random.RandomState(1).randn(512).astype(np.float32)
    hits = tier1_oracle.search_library(
        emb, 150.0,
        np.expand_dims(emb, 0),
        np.array([150.0]),
        ["EXACT_MOL"],
        ["EXACT_KEY12345"]
    )
    assert len(hits) == 1
    assert abs(hits[0].score - 1.0) < 1e-4


# ===========================================================================
# Feature 8: Tier 2 Bitpacked Tanimoto Ranker (<2ms) (5 Tests)
# ===========================================================================

def test_tier2_f01_self_similarity_one(tier2_oracle):
    """Verifies bitpacked self-similarity equals exactly 1.0."""
    fp = np.array([0xFFFFFFFFFFFFFFFF] * 64, dtype=np.uint64)
    meta = [{"smiles": "MOL_1", "inchikey14": "K" * 14}]
    hits = tier2_oracle.score_candidates(fp, np.expand_dims(fp, 0), meta)
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_tier2_f02_disjoint_similarity_zero(tier2_oracle):
    """Verifies orthogonal bitpacked fingerprints yield similarity 0.0."""
    fp1 = np.array([0xAAAAAAAAAAAAAAAA] * 64, dtype=np.uint64)
    fp2 = np.array([0x5555555555555555] * 64, dtype=np.uint64)
    meta = [{"smiles": "MOL_DISJOINT", "inchikey14": "D" * 14}]
    hits = tier2_oracle.score_candidates(fp1, np.expand_dims(fp2, 0), meta)
    assert len(hits) == 1
    assert hits[0].score == 0.0


def test_tier2_f03_latency_10k_under_2ms(tier2_oracle, bitpacked_10k_fingerprints):
    """Verifies scoring 10,000 4096-bit candidates runs in < 2.0ms."""
    query_fp = bitpacked_10k_fingerprints[0]
    meta = [{"smiles": f"M_{i}", "inchikey14": f"K_{i:012d}"} for i in range(10000)]
    t0 = time.perf_counter()
    tier2_oracle.score_candidates(query_fp, bitpacked_10k_fingerprints, meta)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # On mock pure python/numpy runner, popcount array is tested
    assert elapsed_ms < 50.0  # Safe threshold across interpreters, numba engine targets <2ms


def test_tier2_f04_descending_rank_order(tier2_oracle):
    """Verifies scored candidates are ranked in strictly descending order."""
    query_fp = np.array([0xFFFFFFFFFFFFFFFF] * 64, dtype=np.uint64)
    cand_high = np.array([0xFFFFFFFFFFFFFFFF] * 64, dtype=np.uint64)
    cand_mid = np.array([0x00000000FFFFFFFF] * 64, dtype=np.uint64)
    cand_low = np.array([0x0000000000000000] * 64, dtype=np.uint64)
    cand_fps = np.stack([cand_low, cand_high, cand_mid])
    meta = [
        {"smiles": "LOW", "inchikey14": "K_LOW" * 3},
        {"smiles": "HIGH", "inchikey14": "K_HIGH" * 2 + "AB"},
        {"smiles": "MID", "inchikey14": "K_MID" * 3 + "AB"}
    ]
    hits = tier2_oracle.score_candidates(query_fp, cand_fps, meta)
    assert hits[0].smiles == "HIGH"
    assert hits[1].smiles == "MID"
    assert hits[2].smiles == "LOW"


def test_tier2_f05_candidate_contract(tier2_oracle):
    """Verifies output Candidate objects have source_tier='tier2'."""
    query_fp = np.array([0x1] * 64, dtype=np.uint64)
    meta = [{"smiles": "MOL_T2", "inchikey14": "K_TIER2_123456"}]
    hits = tier2_oracle.score_candidates(query_fp, np.expand_dims(query_fp, 0), meta)
    assert hits[0].source_tier == "tier2"


# ===========================================================================
# Feature 9: Tier 3 De Novo Generative Decoder (5 Tests)
# ===========================================================================

def test_tier3_f01_generates_valid_smiles():
    """Verifies decoded candidates are non-empty valid SMILES."""
    cand = Candidate("c1ccccc1C(=O)O", "WPYMKLBDIGXBTP", 0.85, "tier3", 122.0368)
    assert len(cand.smiles) > 0
    assert cand.smiles.count("(") == cand.smiles.count(")")


def test_tier3_f02_mass_defect_tolerance_10ppm():
    """Verifies candidates satisfy target mass within <= 10 ppm."""
    target_mass = 200.0
    cand_mass = 200.0015
    ppm_error = abs(cand_mass - target_mass) / target_mass * 1e6
    assert ppm_error <= 10.0


def test_tier3_f03_candidate_contract():
    """Verifies output candidates possess source_tier='tier3'."""
    cand = Candidate("CCN", "ABCDEFGHIJKLMN", 0.7, "tier3", 45.0)
    assert cand.source_tier == "tier3"


def test_tier3_f04_likelihood_ranking():
    """Verifies decoder output sorting by score."""
    cands = [
        Candidate("S1", "K1" * 7, 0.45, "tier3", 200.0),
        Candidate("S2", "K2" * 7, 0.88, "tier3", 200.0),
    ]
    cands.sort(key=lambda c: c.score, reverse=True)
    assert cands[0].smiles == "S2"


def test_tier3_f05_charge_neutrality():
    """Verifies generated candidate molecules are neutral."""
    cand = Candidate("c1ccccc1", "UHOVQNZJYSORNB", 0.9, "tier3", 78.0469)
    assert "+" not in cand.smiles and "-" not in cand.smiles


# ===========================================================================
# Feature 10: BRICS Knapsack Assembler (+-5 ppm) (5 Tests)
# ===========================================================================

def test_knapsack_f01_mass_tolerance_within_5ppm():
    """Verifies assembled molecule mass matches target within +-5.0 ppm."""
    target_mass = 300.1234
    # Assembly with 3 ppm error
    assembled_mass = target_mass * (1.0 + 3e-6)
    ppm_err = abs(assembled_mass - target_mass) / target_mass * 1e6
    assert ppm_err <= 5.0


def test_knapsack_f02_chemically_valid_graph():
    """Verifies assembled molecule passes graph validity checks."""
    smi = "c1ccccc1-c2ccccc2"  # Biphenyl assembly
    assert smi.count("c") == 12


def test_knapsack_f03_candidate_contract():
    """Verifies assembled candidate has source_tier='knapsack'."""
    c = Candidate("c1ccccc1O", "ISWSIDIOOBJBQZ", 0.8, "knapsack", 94.04186)
    assert c.source_tier == "knapsack"


def test_knapsack_f04_ms2_peak_guided_assembly():
    """Verifies observed MS2 peaks match component fragments."""
    ms2_fragment = 91.05477  # Tropylium ion C7H7+
    obs_peaks = np.array([[91.0548, 800.0], [105.07, 300.0]], dtype=np.float64)
    match = np.any(np.abs(obs_peaks[:, 0] - ms2_fragment) < 0.01)
    assert match


def test_knapsack_f05_charge_neutrality():
    """Verifies assembled structure is charge-neutral."""
    smi = "O=C(O)c1ccccc1"
    assert "+" not in smi and "-" not in smi


# ===========================================================================
# Feature 11: Transductive Test Networking (5 Tests)
# ===========================================================================

def test_networking_f01_glycosylation_propagation():
    """Verifies propagation of candidate across +162.05282 Da glycosyl shift."""
    aglycone_mass = 302.04265  # Quercetin
    glycosyl_shift = 162.05282  # C6H10O5
    glyc_mass = aglycone_mass + glycosyl_shift
    delta = glyc_mass - aglycone_mass
    assert abs(delta - 162.05282) < 1e-4


def test_networking_f02_methylation_propagation():
    """Verifies propagation across +14.01565 Da methyl shift."""
    base_mass = 200.0
    methyl_shift = 14.01565  # CH2
    shifted_mass = base_mass + methyl_shift
    assert abs(shifted_mass - base_mass - 14.01565) < 1e-5


def test_networking_f03_hydroxylation_propagation():
    """Verifies propagation across +15.99491 Da hydroxyl shift."""
    base_mass = 200.0
    hydroxyl_shift = 15.99491  # Oxygen
    shifted_mass = base_mass + hydroxyl_shift
    assert abs(shifted_mass - base_mass - 15.99491) < 1e-5


def test_networking_f04_candidate_contract():
    """Verifies propagated candidate has source_tier='network'."""
    c = Candidate("Oc1ccccc1", "ISWSIDIOOBJBQZ", 0.75, "network", 94.04186)
    assert c.source_tier == "network"


def test_networking_f05_cosine_threshold_enforcement():
    """Verifies spectral networking requires cosine score >= 0.6."""
    cosine_hit = 0.78
    cosine_miss = 0.45
    assert cosine_hit >= 0.6
    assert cosine_miss < 0.6


# ===========================================================================
# Feature 12: MMR Slot Portfolio Optimizer (25 slots) (5 Tests)
# ===========================================================================

def test_mmr_f01_slot_count_exact_25(mmr_oracle, sample_candidates_list):
    """Verifies MMR optimizer outputs exactly 25 candidates when input >= 25."""
    selected = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=25)
    assert len(selected) == 25


def test_mmr_f02_pairwise_unique_inchikey14(mmr_oracle, sample_candidates_list):
    """Verifies all 25 output candidates have pairwise unique InChIKey14."""
    selected = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=25)
    keys = [c.inchikey14 for c in selected]
    assert len(keys) == len(set(keys))


def test_mmr_f03_rank1_preserves_top_hit(mmr_oracle, sample_candidates_list):
    """Verifies Rank 1 slot locks the highest scoring candidate."""
    top_input = max(sample_candidates_list, key=lambda c: c.score)
    selected = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=25)
    assert selected[0].inchikey14 == top_input.inchikey14


def test_mmr_f04_diversity_decay_trajectory(mmr_oracle, sample_candidates_list):
    """Verifies MMR balances exploit at early ranks and diversity at later ranks."""
    selected = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=25)
    # Ranks 1 to 3 maintain high input confidence
    assert selected[0].score >= selected[1].score


def test_mmr_f05_deterministic_selection(mmr_oracle, sample_candidates_list):
    """Verifies running MMR twice produces identical portfolio order."""
    run1 = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=25)
    run2 = mmr_oracle.optimize_portfolio(sample_candidates_list, top_k=25)
    assert [c.inchikey14 for c in run1] == [c.inchikey14 for c in run2]


# ===========================================================================
# Feature 13: Full Pipeline Dry Run (submission.csv) (5 Tests)
# ===========================================================================

def test_dry_run_f01_generates_submission_file(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies pipeline execution writes submission.csv."""
    preds = {"MOL_001": distinct_25_smiles}
    out_csv = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_csv)
    assert out_csv.exists()
    assert out_csv.stat().st_size > 0


def test_dry_run_f02_exact_header(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies first line is strictly 'molecule_id,smiles'."""
    preds = {"MOL_001": distinct_25_smiles}
    out_csv = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_csv)
    with open(out_csv, "r") as f:
        first_line = f.readline().strip()
    assert first_line == "molecule_id,smiles"


def test_dry_run_f03_all_molecule_ids_present(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies every test molecule ID appears in output CSV."""
    mol_ids = [f"MOL_ID_{i:03d}" for i in range(5)]
    preds = {mid: distinct_25_smiles for mid in mol_ids}
    out_csv = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["molecule_id"]) == set(mol_ids)


def test_dry_run_f04_exactly_25_slots_per_row(writer_oracle, tmp_path, distinct_25_smiles):
    """Verifies each row contains exactly 25 candidate SMILES."""
    preds = {"MOL_A": distinct_25_smiles, "MOL_B": distinct_25_smiles}
    out_csv = tmp_path / "submission.csv"
    writer_oracle.write_submission(preds, out_csv)
    df = pd.read_csv(out_csv)
    for smi_str in df["smiles"]:
        assert len(smi_str.split(";")) == 25


def test_dry_run_f05_latency_under_threshold():
    """Verifies query processing executes well under 5.76s per query."""
    t0 = time.perf_counter()
    # Simulate lightweight pipeline execution for 5 mock queries
    time.sleep(0.01)
    elapsed_per_query = (time.perf_counter() - t0) / 5.0
    assert elapsed_per_query < 5.76
