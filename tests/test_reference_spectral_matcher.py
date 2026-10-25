import numpy as np
import pytest
from src.retrieval.reference_spectral_matcher import ReferenceSpectralMatcher, fast_spectral_cosine


def test_fast_spectral_cosine_exact():
    peaks = np.array([[100.0, 10.0], [200.0, 50.0], [300.0, 100.0]])
    score = fast_spectral_cosine(peaks, peaks)
    assert abs(score - 1.0) < 1e-4


def test_fast_spectral_cosine_disjoint():
    p1 = np.array([[100.0, 10.0]])
    p2 = np.array([[500.0, 10.0]])
    score = fast_spectral_cosine(p1, p2)
    assert score == 0.0


def test_fast_spectral_cosine_empty():
    p1 = np.zeros((0, 2))
    p2 = np.array([[100.0, 10.0]])
    assert fast_spectral_cosine(p1, p2) == 0.0
    assert fast_spectral_cosine(None, p2) == 0.0


def test_reference_spectral_matcher_no_path():
    matcher = ReferenceSpectralMatcher(train_path=None)
    assert matcher.is_available() is False
    res = matcher.match_query(300.0, np.array([[100.0, 1.0]]))
    assert res == []


def test_reference_spectral_matcher_mock_data():
    matcher = ReferenceSpectralMatcher(train_path=None)
    # Populate mock in-memory reference arrays
    matcher.ref_mzs = np.array([200.0, 300.0, 400.0], dtype=np.float64)
    matcher.ref_adducts = np.array(["[M+H]+", "[M+H]+", "[M-H]-"], dtype=object)
    matcher.ref_iks = np.array(["IK14REF0000001", "IK14REF0000002", "IK14REF0000003"], dtype=object)
    matcher.ref_smiles = np.array(["CCO", "c1ccccc1O", "CC(=O)O"], dtype=object)
    matcher.ref_mzs_list = [
        [100.0, 200.0],
        [100.0, 150.0, 300.0],
        [60.0, 120.0],
    ]
    matcher.ref_ints_list = [
        [50.0, 100.0],
        [30.0, 50.0, 100.0],
        [40.0, 100.0],
    ]
    matcher.is_loaded = True

    assert matcher.is_available() is True

    # Query matching mass 300.0 within 10 ppm
    q_peaks = np.array([[100.0, 30.0], [150.0, 50.0], [300.0, 100.0]])
    res = matcher.match_query(
        precursor_mz=300.0,
        peaks=q_peaks,
        adduct="[M+H]+",
        tolerance_ppm=10.0,
        min_cosine=0.50,
    )
    assert len(res) == 1
    assert res[0]["inchikey14"] == "IK14REF0000002"
    assert res[0]["smiles"] == "c1ccccc1O"
    assert res[0]["score"] > 0.99
    assert res[0]["source"] == "track1_dreams"
