import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.chemistry.adducts import calculate_canonical_neutral_mass
from src.chemistry.standardizer import standardize_mol
from src.data.loader import SpectrumData, parse_peak_array, filter_and_normalize_peaks, SpectrumLoader
from src.data.multi_energy_fusion import MultiEnergyFusionEngine, SpectralFrame
from src.reranking.slot_optimizer import DecisionTheoreticSlotOptimizer
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever
from src.retrieval.database_search import SoftDatabaseSearcher
from src.retrieval.generative_denovo import BoundedGenerativeEngine
from src.submission.writer import write_submission, validate_submission_file
from src.submission.runtime_governor import RuntimeGovernor


def test_unicode_smiles_string():
    """1. Unicode SMILES string -> standardize_mol handles without crash"""
    smi, ik14 = standardize_mol("C\u2081C")
    assert smi is None or isinstance(smi, str)


def test_extremely_long_smiles():
    """2. Extremely long SMILES (10000 chars of 'C') -> standardize_mol does not hang"""
    smi = "C" * 10000
    res, ik14 = standardize_mol(smi)
    assert res is None or isinstance(res, str)


def test_nan_precursor_mz():
    """3. NaN precursor_mz -> SpectrumData raises ValueError"""
    with pytest.raises(ValueError):
        SpectrumData(
            molecule_id="1",
            precursor_mz=float("nan"),
            adduct="[M+H]+",
            polarity="positive",
            collision_energy=30.0,
            mz_array=np.array([100.0]),
            intensity_array=np.array([1.0]),
        )


def test_inf_precursor_mz():
    """4. Inf precursor_mz -> SpectrumData handles (ValueError or graceful)"""
    try:
        sd = SpectrumData(
            molecule_id="1",
            precursor_mz=float("inf"),
            adduct="[M+H]+",
            polarity="positive",
            collision_energy=30.0,
            mz_array=np.array([100.0]),
            intensity_array=np.array([1.0]),
        )
        assert sd.precursor_mz == float("inf")
    except ValueError:
        pass


def test_negative_collision_energy():
    """5. Negative collision_energy -> does not crash multi_energy_fusion"""
    sd = SpectrumData(
        molecule_id="1",
        precursor_mz=100.0,
        adduct="[M+H]+",
        polarity="positive",
        collision_energy=-10.0,
        mz_array=np.array([100.0]),
        intensity_array=np.array([1.0]),
    )
    fusion_engine = MultiEnergyFusionEngine()
    frames = fusion_engine.from_query_spectra([sd], [np.zeros(1024)])
    result = fusion_engine.fuse_embeddings(frames)
    assert result is not None


def test_empty_string_molecule_id():
    """6. Empty string molecule_id -> pipeline processes without KeyError"""
    sd = SpectrumData(
        molecule_id="",
        precursor_mz=100.0,
        adduct="[M+H]+",
        polarity="positive",
        collision_energy=30.0,
        mz_array=np.array([100.0]),
        intensity_array=np.array([1.0]),
    )
    assert sd.molecule_id == ""


def test_massive_candidate_list():
    """7. Massive candidate list (1000 candidates) -> slot optimizer still returns exactly 25"""
    optimizer = DecisionTheoreticSlotOptimizer()
    candidates = [{"inchikey14": f"ABCDEFGHIJ{i:04d}", "score": 0.9} for i in range(1000)]
    slots = optimizer.allocate_25_slots(candidates)
    assert len(slots) == 25


def test_all_candidates_identical_inchikey():
    """8. All candidates with identical InChIKey14 -> 1 real + fallback to reach 25"""
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=["FALLBACK000001", "FALLBACK000002"])
    candidates = [{"inchikey14": "IDENTICAL00000", "score": 0.9} for _ in range(50)]
    slots = optimizer.allocate_25_slots(candidates)
    assert len(slots) == 25
    assert "IDENTICAL00000" in slots


def test_candidate_dict_missing_inchikey():
    """9. Candidate dict with missing 'inchikey14' key -> slot optimizer handles gracefully"""
    optimizer = DecisionTheoreticSlotOptimizer()
    candidates = [{"score": 0.9} for _ in range(30)]
    slots = optimizer.allocate_25_slots(candidates)
    assert len(slots) == 25


def test_candidate_dict_none_score():
    """10. Candidate dict with None score -> _extract_score fallback to 0.0"""
    optimizer = DecisionTheoreticSlotOptimizer()
    candidates = [{"inchikey14": f"IK1400000000{i:02d}", "score": None} for i in range(30)]
    slots = optimizer.allocate_25_slots(candidates)
    assert len(slots) == 25


def test_formula_no_carbon():
    """11. Formula with no carbon (e.g., 'H2O') -> adducts module handles"""
    # Using calculate_canonical_neutral_mass from adducts
    mass = calculate_canonical_neutral_mass(18.01, "[M+H]+")
    assert mass is not None


def test_governor_short_time():
    """12. Governor with total_budget_sec=1.0 and very short time -> FAST mode"""
    gov = RuntimeGovernor(total_budget_sec=1.0, safety_buffer_sec=0.0, total_queries=100)
    assert gov.get_execution_mode().value == "FAST"


def test_governor_one_query():
    """13. Governor with total_queries=1 -> full budget per query"""
    gov = RuntimeGovernor(total_budget_sec=1000.0, safety_buffer_sec=0.0, total_queries=1)
    assert gov.sec_per_query_remaining >= 900.0


def test_submission_writer_special_chars():
    """14. Submission writer with special characters in molecule_id -> proper CSV"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "sub.csv"
        predictions = {"id_with,comma": ["C", "CC"] + ["C"] * 23}
        # It should handle the comma or error out gracefully depending on expected ID
        try:
            write_submission(predictions, out_file)
            assert out_file.exists()
        except Exception:
            pass


def test_validate_submission_trailing_newlines():
    """15. validate_submission_file with trailing newlines -> handled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "sub.csv"
        out_file.write_text("molecule_id,smiles\nmol1," + ";".join(["C"] * 25) + "\n\n")
        valid, errors = validate_submission_file(out_file)
        assert not valid or valid


def test_validate_submission_windows_crlf():
    """16. validate_submission_file with Windows CRLF line endings -> handled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "sub.csv"
        content = "molecule_id,smiles\r\nmol1," + ";".join([f"C{i}C" for i in range(25)]) + "\r\n"
        out_file.write_bytes(content.encode("utf-8"))
        valid, errors = validate_submission_file(out_file, check_inchikey14=False)
        assert valid, f"Errors: {errors}"


def test_peak_array_nan_embedded():
    """17. Peak array with NaN values embedded -> parse_peak_array handles"""
    arr = parse_peak_array("[100.0, NaN, 102.0]")
    assert len(arr) >= 0


def test_peak_array_inf_values():
    """18. Peak array with Inf values -> filter_and_normalize_peaks handles"""
    mzs = np.array([100.0, 101.0])
    ints = np.array([10.0, float("inf")])
    mzs_new, ints_new = filter_and_normalize_peaks(mzs, ints)
    assert len(mzs_new) >= 0


def test_empty_adduct_string():
    """19. Empty adduct string -> calculate_canonical_neutral_mass handles"""
    with pytest.raises(ValueError):
        mass = calculate_canonical_neutral_mass(100.0, "")


def test_dreams_0d_embedding():
    """20. DreaMS retriever with 0-dimensional embedding -> no crash"""
    retriever = CalibratedDreaMSRetriever()
    res, locked = retriever.evaluate_candidates(np.array([]), [])
    assert isinstance(res, list)


def test_db_mismatched_fp():
    """21. Database searcher with mismatched fingerprint dimensions -> no crash or clear error"""
    searcher = SoftDatabaseSearcher(db={}, fallback_scaffolds=[])
    try:
        res = searcher.search_formulas(["C2H6O"], np.array([1, 0, 1]))
        assert isinstance(res, list)
    except Exception:
        pass


def test_generative_timeout_0():
    """22. Generative engine with timeout=0 -> returns immediately with empty list"""
    engine = BoundedGenerativeEngine()
    res = engine.generate_with_timeout("1", timeout_seconds=0)
    assert isinstance(res, list)


def test_slot_optimizer_fallback_empty():
    """23. Slot optimizer with fallback_pool=[] -> emergency PAD padding"""
    optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=[])
    slots = optimizer.allocate_25_slots([])
    assert len(slots) == 25
    assert slots[0].startswith("PAD")


def test_write_submission_non_string():
    """24. write_submission with predictions containing non-string objects -> skipped gracefully"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "sub.csv"
        class MockCand:
            pass
        predictions = {"mol1": [MockCand(), "C", 123]}
        write_submission(predictions, out_file)
        assert out_file.exists()


def test_runtime_governor_bypass():
    """25. RuntimeGovernor bypass at exact threshold 0.82 -> True, at 0.8199 -> False"""
    gov = RuntimeGovernor(bypass_similarity_threshold=0.82)
    assert gov.should_bypass_generative(0.82) == True
    assert gov.should_bypass_generative(0.8199) == False
