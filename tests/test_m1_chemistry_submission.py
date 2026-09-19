"""
Unit test suite for Milestone 1 (M1: Chemistry Foundation & Submission Safety).

Covers:
1. All 10 competition adduct delta mass calculations, polarity, and roundtrips.
2. Quadrupole isolation +1 13C offset detection, correction, and hypothesis ranking.
3. RDKit salt stripping, charge neutralization, tautomer canonicalization, and InChIKey14 extraction.
4. Atomic CSV submission writer (25 slots, semicolon-delimited, atomic replace, valid format).
5. Runtime governor regime transitions, time tracking, module timeouts, and telemetry export.
6. Synthetic mock data generation and Parquet / CSV roundtrip loading via SpectrumLoader.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.chemistry.adducts import (
    CARBON_13_DELTA,
    COMPETITION_ADDUCTS,
    ELECTRON_MASS,
    PROTON_MASS,
    calculate_neutral_mass,
    calculate_precursor_mz,
    correct_precursor_mz,
    detect_ms2_precursor_cluster,
    get_adduct_candidates,
    get_adduct_info,
    get_precursor_hypotheses,
    normalize_adduct_name,
)
from src.chemistry.standardizer import (
    deduplicate_candidates,
    deduplicate_smiles_list,
    get_inchikey14,
    standardize_mol,
)
from src.submission.writer import (
    DEFAULT_FALLBACK_SMILES,
    IncompleteSubmissionError,
    SubmissionValidationError,
    validate_submission_file,
    write_submission,
)
from src.submission.runtime_governor import (
    OperatingRegime,
    RuntimeGovernor,
)
from src.data.loader import (
    QuerySpectrum,
    SpectrumData,
    SpectrumLoader,
    filter_and_normalize_peaks,
    parse_peak_array,
)
from src.data.mock_data import (
    ARCHETYPE_COMPOUNDS,
    MockDataGenerator,
)

# Reference known natural product: Quercetin (C15H10O7)
QUERCETIN_MASS = 302.042653


# =====================================================================
# 1. Adduct Delta Masses & Precursor Conversion Tests
# =====================================================================

class TestCompetitionAdducts:
    """Verifies all 10 competition adducts, formulas, deltas, and polarities."""

    @pytest.mark.parametrize(
        "adduct,expected_delta,expected_polarity,expected_charge",
        [
            ("[M+H]+", 1.007276452, "positive", 1),
            ("[M+NH4]+", 18.033825553, "positive", 1),
            ("[M+Na]+", 22.989220702, "positive", 1),
            ("[M+K]+", 38.963157906, "positive", 1),
            ("[M-H2O+H]+", -17.003288232, "positive", 1),
            ("[M-H]-", -1.007276452, "negative", -1),
            ("[M+Cl]-", 34.969401301, "negative", -1),
            ("[M+FA-H]-", 44.998202851, "negative", -1),
            ("[M+Hac-H]-", 59.013852916, "negative", -1),
            ("[M-H2O-H]-", -19.017841136, "negative", -1),
        ],
    )
    def test_all_10_competition_adduct_properties(
        self, adduct: str, expected_delta: float, expected_polarity: str, expected_charge: int
    ):
        info = get_adduct_info(adduct)
        assert info.name == adduct
        assert pytest.approx(info.delta_mass, abs=1e-8) == expected_delta
        assert info.polarity == expected_polarity
        assert info.charge == expected_charge
        assert info.mult == 1

    @pytest.mark.parametrize("adduct", list(COMPETITION_ADDUCTS.keys()))
    def test_adduct_round_trip_conversion(self, adduct: str):
        mz = calculate_precursor_mz(QUERCETIN_MASS, adduct)
        recovered_mass = calculate_neutral_mass(mz, adduct)
        assert pytest.approx(recovered_mass, abs=1e-8) == QUERCETIN_MASS

    def test_get_adduct_candidates_positive_mode(self):
        pos_candidates = get_adduct_candidates(303.05, "positive")
        assert len(pos_candidates) == 5
        names = [a for a, _ in pos_candidates]
        assert "[M+H]+" in names
        assert "[M+NH4]+" in names
        assert "[M+Na]+" in names
        assert "[M+K]+" in names
        assert "[M-H2O+H]+" in names
        # Ensure all calculated neutral masses are strictly positive
        for _, mass in pos_candidates:
            assert mass > 0.0

    def test_get_adduct_candidates_negative_mode(self):
        neg_candidates = get_adduct_candidates(301.03, "negative")
        assert len(neg_candidates) == 5
        names = [a for a, _ in neg_candidates]
        assert "[M-H]-" in names
        assert "[M+Cl]-" in names
        assert "[M+FA-H]-" in names
        assert "[M+Hac-H]-" in names
        assert "[M-H2O-H]-" in names
        for _, mass in neg_candidates:
            assert mass > 0.0

    def test_adduct_alias_normalization(self):
        assert normalize_adduct_name("[M+HCOO]-") == "[M+FA-H]-"
        assert normalize_adduct_name("[M+CH3COO]-") == "[M+Hac-H]-"
        assert normalize_adduct_name("[M+OAc]-") == "[M+Hac-H]-"
        assert normalize_adduct_name("[M+H-H2O]+") == "[M-H2O+H]+"
        assert normalize_adduct_name("[M-H-H2O]-") == "[M-H2O-H]-"
        assert normalize_adduct_name(" [M + H]+ ") == "[M+H]+"

    def test_invalid_adduct_or_mass_raises_error(self):
        with pytest.raises(ValueError):
            calculate_neutral_mass(300.0, "[M+INVALID]+")
        with pytest.raises(ValueError):
            calculate_neutral_mass(-10.0, "[M+H]+")
        with pytest.raises(ValueError):
            calculate_precursor_mz(-10.0, "[M+H]+")
        with pytest.raises(ValueError):
            get_adduct_candidates(300.0, "unknown_mode")


# =====================================================================
# 2. 13C Offset & Precursor Correction Tests
# =====================================================================

class TestPrecursorCorrection:
    """Verifies +1 13C quadrupole mispick detection and multi-hypothesis ranking."""

    def test_detect_ms2_precursor_cluster_presence(self):
        true_m0 = 303.0499
        mispicked_m1 = true_m0 + CARBON_13_DELTA
        # MS2 scan contains the residual unfragmented M+0 peak
        ms2 = np.array([
            [153.018, 100.0],
            [true_m0, 25.0],       # Residual M+0 peak present
            [mispicked_m1, 40.0],  # Isolated M+1 peak
        ])
        assert detect_ms2_precursor_cluster(mispicked_m1, ms2, tolerance_da=0.02) is True
        # If testing true_m0, there is no peak at true_m0 - 1.003355
        assert detect_ms2_precursor_cluster(true_m0, ms2, tolerance_da=0.02) is False

    def test_correct_precursor_mz_with_and_without_ms2(self):
        true_m0 = 303.0499
        mispicked_m1 = true_m0 + CARBON_13_DELTA
        ms2 = np.array([[true_m0, 30.0], [mispicked_m1, 50.0]])

        # With evidence: corrects to true_m0
        corrected = correct_precursor_mz(mispicked_m1, ms2)
        assert pytest.approx(corrected, abs=1e-4) == true_m0

        # Without evidence or empty ms2: keeps observed
        assert correct_precursor_mz(mispicked_m1, None) == mispicked_m1
        assert correct_precursor_mz(mispicked_m1, np.zeros((0, 2))) == mispicked_m1

    def test_precursor_hypotheses_ranking(self):
        observed = 400.1234
        # Case A: No MS2 evidence -> Nominal is rank 1
        hyps_nominal = get_precursor_hypotheses(observed)
        assert len(hyps_nominal) == 3
        assert hyps_nominal[0][0] == observed
        assert hyps_nominal[0][1] == 1.0
        assert pytest.approx(hyps_nominal[1][0], abs=1e-5) == observed - CARBON_13_DELTA

        # Case B: Confirmed MS2 evidence -> Mispick is rank 1
        ms2 = np.array([[observed - CARBON_13_DELTA, 50.0]])
        hyps_confirmed = get_precursor_hypotheses(observed, ms2)
        assert len(hyps_confirmed) == 2
        assert pytest.approx(hyps_confirmed[0][0], abs=1e-5) == observed - CARBON_13_DELTA
        assert hyps_confirmed[0][1] == 1.0
        assert hyps_confirmed[0][2] == "13C_confirmed_mispick"


# =====================================================================
# 3. RDKit Standardization & InChIKey14 Tests
# =====================================================================

class TestMolecularStandardizer:
    """Verifies salt stripping, neutralization, tautomer canonicalization, and InChIKey14."""

    @dataclass
    class MockCandidate:
        smiles: str
        inchikey14: str
        score: float

    def test_tautomer_canonicalization_collapse(self):
        # 2-pyridone vs 2-hydroxypyridine must produce identical InChIKey14
        smi_a, ik_a = standardize_mol("O=C1NC=CC=C1")
        smi_b, ik_b = standardize_mol("Oc1ncccc1")
        assert ik_a is not None and ik_b is not None
        assert ik_a == ik_b == "UBQKCCHYAOITMY"
        assert smi_a == smi_b == "O=c1cccc[nH]1"

    def test_salt_stripping_and_charge_neutralization(self):
        # Sodium benzoate and neutral benzoic acid must yield identical InChIKey14
        _, ik_salt = standardize_mol("[Na+].[O-]C(=O)c1ccccc1")
        _, ik_acid = standardize_mol("O=C(O)c1ccccc1")
        assert ik_salt is not None and ik_acid is not None
        assert ik_salt == ik_acid == "WPYMKLBDIGXBTP"

    def test_stereoisomer_skeletal_identity(self):
        # D-glucose and L-glucose have different stereochemistry but identical InChIKey14
        d_glucose = "OC[C@@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        l_glucose = "OC[C@H]1O[C@@H](O)[C@@H](O)[C@H](O)[C@H]1O"
        _, ik_d = standardize_mol(d_glucose)
        _, ik_l = standardize_mol(l_glucose)
        assert ik_d is not None and ik_l is not None
        assert ik_d == ik_l == "WQZGKKKJIJFFOK"

    def test_zwitterion_neutralization(self):
        # Glycine zwitterion neutralizes to neutral glycine
        smi_zwit, ik_zwit = standardize_mol("[NH3+]CC(=O)[O-]")
        smi_neut, ik_neut = standardize_mol("NCC(=O)O")
        assert ik_zwit is not None and ik_neut is not None
        assert ik_zwit == ik_neut == "DHMQDGOQFOQNFH"
        assert smi_zwit == smi_neut == "NCC(=O)O"

    def test_invalid_smiles_handling(self):
        assert standardize_mol("invalid_chem_string") == (None, None)
        assert standardize_mol("C1CC") == (None, None)  # Unclosed ring
        assert standardize_mol("") == (None, None)
        assert standardize_mol(None) == (None, None)

    def test_candidate_deduplication(self):
        c1 = self.MockCandidate("O=C(O)c1ccccc1", "WPYMKLBDIGXBTP", 0.95)
        c2 = self.MockCandidate("[Na+].[O-]C(=O)c1ccccc1", "WPYMKLBDIGXBTP", 0.85)  # Duplicate skeleton
        c3 = self.MockCandidate("Oc1ccccc1", "ISWSIDIOOBJBQZ", 0.75)                # Distinct skeleton

        deduped = deduplicate_candidates([c1, c2, c3])
        assert len(deduped) == 2
        assert deduped[0].smiles == "O=C(O)c1ccccc1"
        assert deduped[1].smiles == "Oc1ccccc1"

    def test_deduplicate_smiles_list(self):
        raw = ["O=C1NC=CC=C1", "Oc1ncccc1", "CC(=O)O", "CC(=O)O"]
        out = deduplicate_smiles_list(raw)
        assert len(out) == 2  # 1 pyridone, 1 acetic acid
        assert out[0][1] == "UBQKCCHYAOITMY"
        assert out[1][1] == "QTBSBXVTEAMEQO"


# =====================================================================
# 4. Atomic Submission CSV Writer Tests
# =====================================================================

class TestSubmissionWriter:
    """Verifies atomic CSV write, InChIKey14 slot validation, backfill and truncation."""

    def test_compliant_submission_write_and_validation(self, tmp_path: Path):
        out_csv = tmp_path / "submission.csv"
        # Provide exactly 25 distinct valid SMILES for 2 test molecules
        predictions = {
            "MOL_0001": DEFAULT_FALLBACK_SMILES[:25],
            "MOL_0002": DEFAULT_FALLBACK_SMILES[:25],
        }

        result_path = write_submission(predictions, out_csv, expected_ids=["MOL_0001", "MOL_0002"])
        assert result_path == out_csv
        assert out_csv.exists()

        is_valid, errors = validate_submission_file(out_csv, expected_ids=["MOL_0001", "MOL_0002"])
        assert is_valid is True, f"Validation failed: {errors}"
        assert len(errors) == 0

    def test_auto_backfill_for_sparse_candidates(self, tmp_path: Path):
        out_csv = tmp_path / "sparse_submission.csv"
        # Only 2 candidates provided for MOL_0001
        predictions = {
            "MOL_0001": ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        }

        write_submission(predictions, out_csv, auto_backfill=True)
        is_valid, errors = validate_submission_file(out_csv, check_inchikey14=True)
        assert is_valid is True, f"Backfill failed: {errors}"

        # Confirm exactly 25 candidates exist on the row
        with open(out_csv, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2  # Header + 1 row
        row_mol_id, cands_str = lines[1].split(",", 1)
        assert row_mol_id == "MOL_0001"
        cands = cands_str.split(";")
        assert len(cands) == 25
        # Ensure our 2 original candidates are at positions 0 and 1
        assert cands[0] == "CC(=O)Oc1ccccc1C(=O)O"
        assert cands[1] == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    def test_auto_truncate_for_excess_candidates(self, tmp_path: Path):
        out_csv = tmp_path / "excess_submission.csv"
        # 30 candidates provided
        excess_cands = DEFAULT_FALLBACK_SMILES[:25] + ["CCCCCO", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC"]
        predictions = {"MOL_0001": excess_cands}

        write_submission(predictions, out_csv, auto_truncate=True)
        is_valid, errors = validate_submission_file(out_csv)
        assert is_valid is True

        with open(out_csv, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        cands = lines[1].split(",", 1)[1].split(";")
        assert len(cands) == 25

    def test_duplicate_inchikey14_rejected_when_validation_enabled(self, tmp_path: Path):
        out_csv = tmp_path / "duplicate_ik14.csv"
        # 2-pyridone and 2-hydroxypyridine have identical InChIKey14
        bad_predictions = {
            "MOL_0001": ["O=C1NC=CC=C1", "Oc1ncccc1"] + DEFAULT_FALLBACK_SMILES[:23]
        }
        # write_submission with validate_inchikey14=True will deduplicate the second one
        # and backfill with an extra unique molecule to maintain 25 unique slots!
        write_submission(bad_predictions, out_csv, validate_inchikey14=True, auto_backfill=True)
        is_valid, errors = validate_submission_file(out_csv, check_inchikey14=True)
        assert is_valid is True

    def test_missing_expected_ids_raises_incomplete_submission_error(self, tmp_path: Path):
        out_csv = tmp_path / "incomplete.csv"
        predictions = {"MOL_0001": DEFAULT_FALLBACK_SMILES[:25]}
        with pytest.raises(IncompleteSubmissionError):
            write_submission(predictions, out_csv, expected_ids=["MOL_0001", "MOL_0002"], auto_backfill=False)

    def test_atomic_write_leaves_no_temporary_files(self, tmp_path: Path):
        out_csv = tmp_path / "atomic_check.csv"
        predictions = {"MOL_0001": DEFAULT_FALLBACK_SMILES[:25]}
        write_submission(predictions, out_csv)
        assert out_csv.exists()

        # Confirm no temporary .tmp files exist in destination directory
        tmp_files = list(tmp_path.glob(".*.tmp"))
        assert len(tmp_files) == 0


# =====================================================================
# 5. Adaptive Runtime Governor Tests
# =====================================================================

class TestRuntimeGovernor:
    """Verifies 3 operating regimes, golden library bypass, and telemetry."""

    def test_regime_transitions_based_on_rate(self):
        # Scenario 1: Abundant time (>5.0s per query) -> DEEP
        gov_deep = RuntimeGovernor(total_budget_sec=6000.0, safety_buffer_sec=100.0, total_queries=100)
        assert gov_deep.sec_per_query_remaining > 5.0
        assert gov_deep.get_execution_mode() == OperatingRegime.DEEP

        # Scenario 2: Moderate time (2.0 - 5.0s per query) -> STANDARD
        gov_std = RuntimeGovernor(total_budget_sec=350.0, safety_buffer_sec=50.0, total_queries=100)
        assert 2.0 <= gov_std.sec_per_query_remaining <= 5.0
        assert gov_std.get_execution_mode() == OperatingRegime.STANDARD

        # Scenario 3: Constrained time (<2.0s per query) -> FAST
        gov_fast = RuntimeGovernor(total_budget_sec=150.0, safety_buffer_sec=50.0, total_queries=100)
        assert gov_fast.sec_per_query_remaining < 2.0
        assert gov_fast.get_execution_mode() == OperatingRegime.FAST

    def test_golden_library_hit_bypass_trigger(self):
        gov = RuntimeGovernor(bypass_similarity_threshold=0.82)
        # Below threshold: do not bypass
        assert gov.should_bypass_generative(0.80) is False
        # At or above threshold: bypass expensive de novo sampling
        assert gov.should_bypass_generative(0.82) is True
        assert gov.should_bypass_generative(0.95) is True

    def test_query_scope_and_module_scope_telemetry(self, tmp_path: Path):
        gov = RuntimeGovernor(total_budget_sec=1000.0, total_queries=2)

        with gov.query_scope("MOL_TEST_1") as regime:
            assert regime in [OperatingRegime.DEEP, OperatingRegime.STANDARD, OperatingRegime.FAST]
            with gov.module_scope("tier1") as timeout:
                assert timeout > 0.0
            with gov.module_scope("tier2") as timeout:
                assert timeout > 0.0

        assert gov.completed_queries == 1
        assert len(gov.telemetry_records) == 1
        rec = gov.telemetry_records[0]
        assert rec.molecule_id == "MOL_TEST_1"
        assert "tier1" in rec.module_durations
        assert "tier2" in rec.module_durations

        # Test telemetry export
        json_path = tmp_path / "telemetry.json"
        gov.export_telemetry(json_path)
        assert json_path.exists()

        with open(json_path, "r") as f:
            data = json.load(f)
        assert "summary" in data
        assert "records" in data
        assert data["summary"]["completed_queries"] == 1


# =====================================================================
# 6. Mock Data & Loader Ingestion Tests
# =====================================================================

class TestDataAndLoader:
    """Verifies synthetic dataset generation, peak filtering, and Parquet/CSV roundtrips."""

    def test_mock_generator_creates_valid_spectra(self):
        generator = MockDataGenerator(seed=123)
        spec = generator.generate_spectrum(compound_idx=0, adduct="[M+H]+")
        assert isinstance(spec, SpectrumData)
        assert spec.precursor_mz > 100.0
        assert spec.polarity == "positive"
        assert len(spec.mz_array) > 0
        assert len(spec.mz_array) == len(spec.intensity_array)
        assert spec.ms2_peaks.shape[1] == 2
        # Base peak normalized to 100
        assert pytest.approx(float(np.max(spec.intensity_array)), abs=1e-3) == 100.0

    def test_mock_generator_13c_mispick_spectral_evidence(self):
        generator = MockDataGenerator(seed=456)
        spec = generator.generate_spectrum(compound_idx=1, adduct="[M+H]+", is_13c_mispick=True)
        assert spec.is_13c_mispick is True
        # Verify MS2 precursor cluster detector detects residual M_0 peak
        assert detect_ms2_precursor_cluster(spec.precursor_mz, spec.ms2_peaks) is True

    def test_mock_library_generation(self):
        generator = MockDataGenerator(seed=789)
        lib_df, embeddings = generator.generate_mock_library(n_entries=10, embedding_dim=128)
        assert len(lib_df) == 10
        assert embeddings.shape == (10, 128)
        # Unit norm check
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_mock_candidate_db_generation(self):
        generator = MockDataGenerator(seed=999)
        db = generator.generate_mock_candidate_db(n_candidates=100, n_bits=4096)
        cand_df = db["candidate_df"]
        fps = db["fingerprints"]
        assert len(cand_df) == 100
        assert fps.shape == (100, 64)  # 4096 bits = 64 uint64
        assert fps.dtype == np.uint64

    def test_peak_filtering_and_normalization(self):
        mzs = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
        ints = np.array([0.001, 10.0, 100.0, 0.0001, 50.0])
        clean_mzs, clean_ints = filter_and_normalize_peaks(mzs, ints, min_rel_intensity=0.01, max_peaks=3)
        assert len(clean_mzs) == 3
        # Should retain top 3 (10.0, 50.0, 100.0) sorted ascending by m/z
        assert clean_mzs.tolist() == [100.0, 150.0, 250.0]
        # Base peak normalized to 100.0
        assert clean_ints[1] == 100.0  # 150.0 was 100.0 originally

    def test_parquet_roundtrip_loading(self, tmp_path: Path):
        generator = MockDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=5)

        pq_file = tmp_path / "test_spectra.parquet"
        SpectrumLoader.save_parquet(dataset, pq_file)
        assert pq_file.exists()

        loaded = SpectrumLoader.load_parquet(pq_file)
        assert len(loaded) == 5
        for orig, roundtrip in zip(dataset, loaded):
            assert orig.molecule_id == roundtrip.molecule_id
            assert pytest.approx(orig.precursor_mz, abs=1e-6) == roundtrip.precursor_mz
            assert orig.adduct == roundtrip.adduct
            assert orig.polarity == roundtrip.polarity
            assert len(orig.mz_array) == len(roundtrip.mz_array)
            assert np.allclose(orig.mz_array, roundtrip.mz_array, atol=1e-5)

    def test_csv_roundtrip_loading(self, tmp_path: Path):
        generator = MockDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=5)

        csv_file = tmp_path / "test_spectra.csv"
        SpectrumLoader.save_csv(dataset, csv_file)
        assert csv_file.exists()

        loaded = SpectrumLoader.load_csv(csv_file)
        assert len(loaded) == 5
        for orig, roundtrip in zip(dataset, loaded):
            assert orig.molecule_id == roundtrip.molecule_id
            assert pytest.approx(orig.precursor_mz, abs=1e-6) == roundtrip.precursor_mz
            assert orig.adduct == roundtrip.adduct
            assert np.allclose(orig.mz_array, roundtrip.mz_array, atol=1e-5)

    def test_to_query_spectra_conversion(self):
        generator = MockDataGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=3)
        query_specs = SpectrumLoader.to_query_spectra(dataset)
        assert len(query_specs) == 3
        for q in query_specs:
            assert isinstance(q, QuerySpectrum)
            assert isinstance(q.mz_array, list)
            assert isinstance(q.intensity_array, list)
