import pytest
import numpy as np
import pandas as pd
import math
from pathlib import Path

from src.data.loader import (
    parse_peak_array,
    filter_and_normalize_peaks,
    SpectrumData,
    SpectrumLoader,
    QuerySpectrum
)
from src.data.mock_data import MockDataGenerator

# parse_peak_array
def test_parse_peak_array_none():
    res = parse_peak_array(None)
    assert isinstance(res, np.ndarray)
    assert len(res) == 0

def test_parse_peak_array_nan():
    res = parse_peak_array(float('nan'))
    assert isinstance(res, np.ndarray)
    assert len(res) == 0

def test_parse_peak_array_empty_string():
    res = parse_peak_array("")
    assert isinstance(res, np.ndarray)
    assert len(res) == 0

def test_parse_peak_array_whitespace_string():
    res = parse_peak_array("   ")
    assert isinstance(res, np.ndarray)
    assert len(res) == 0

def test_parse_peak_array_json_list():
    res = parse_peak_array("[1.0, 2.0, 3.0]")
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, [1.0, 2.0, 3.0])

def test_parse_peak_array_whitespace_delimited():
    res = parse_peak_array("1.0 2.0 3.0")
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, [1.0, 2.0, 3.0])

def test_parse_peak_array_semicolon_delimited():
    res = parse_peak_array("1.0;2.0;3.0")
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, [1.0, 2.0, 3.0])

def test_parse_peak_array_comma_delimited():
    res = parse_peak_array("1.0,2.0,3.0")
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, [1.0, 2.0, 3.0])

def test_parse_peak_array_list():
    res = parse_peak_array([1.0, 2.0])
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, [1.0, 2.0])

def test_parse_peak_array_ndarray():
    arr = np.array([1.0])
    res = parse_peak_array(arr)
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, [1.0])

# filter_and_normalize_peaks
def test_filter_and_normalize_peaks_empty():
    mz, ints = filter_and_normalize_peaks(np.array([]), np.array([]))
    assert len(mz) == 0
    assert len(ints) == 0

def test_filter_and_normalize_peaks_all_zero():
    mz, ints = filter_and_normalize_peaks(np.array([100.0, 200.0]), np.array([0.0, 0.0]))
    assert len(mz) == 0
    assert len(ints) == 0

def test_filter_and_normalize_peaks_single_peak():
    mz, ints = filter_and_normalize_peaks(np.array([150.0]), np.array([50.0]))
    np.testing.assert_allclose(ints, [100.0])
    np.testing.assert_allclose(mz, [150.0])

def test_filter_and_normalize_peaks_truncate():
    mz_in = np.linspace(100, 200, 150)
    int_in = np.linspace(1, 150, 150)
    mz, ints = filter_and_normalize_peaks(mz_in, int_in, max_peaks=100)
    assert len(mz) == 100
    assert len(ints) == 100
    assert mz[0] == mz_in[50]

def test_filter_and_normalize_peaks_sorted():
    mz_in = np.array([300.0, 100.0, 200.0])
    int_in = np.array([50.0, 10.0, 100.0])
    mz, ints = filter_and_normalize_peaks(mz_in, int_in)
    np.testing.assert_allclose(mz, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(ints, [10.0, 100.0, 50.0])

def test_filter_and_normalize_peaks_min_rel_intensity():
    mz_in = np.array([100.0, 200.0, 300.0])
    int_in = np.array([1000.0, 10.0, 0.5])
    mz, ints = filter_and_normalize_peaks(mz_in, int_in, min_rel_intensity=0.001)
    assert len(mz) == 2
    np.testing.assert_allclose(mz, [100.0, 200.0])
    np.testing.assert_allclose(ints, [100.0, 1.0])

# SpectrumData
def test_spectrumdata_precursor_zero():
    with pytest.raises(ValueError, match="precursor_mz must be strictly positive"):
        SpectrumData(molecule_id="1", precursor_mz=0.0, adduct="[M+H]+", polarity="positive", collision_energy=None, mz_array=[], intensity_array=[])

def test_spectrumdata_mismatch_length():
    with pytest.raises(ValueError, match="Peak array length mismatch"):
        SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=None, mz_array=[1.0], intensity_array=[])

def test_spectrumdata_ms2_peaks():
    sd = SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=None, mz_array=[1.0, 2.0], intensity_array=[10.0, 20.0])
    ms2 = sd.ms2_peaks
    assert ms2.shape == (2, 2)
    np.testing.assert_allclose(ms2, [[1.0, 10.0], [2.0, 20.0]])

def test_spectrumdata_ms2_peaks_empty():
    sd = SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=None, mz_array=[], intensity_array=[])
    ms2 = sd.ms2_peaks
    assert ms2.shape == (0, 2)

def test_spectrumdata_to_query_spectrum():
    sd = SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=20.0, mz_array=[1.0], intensity_array=[10.0])
    qs = sd.to_query_spectrum()
    assert isinstance(qs, QuerySpectrum)
    assert qs.molecule_id == "1"
    assert qs.precursor_mz == 100.0
    assert qs.adduct == "[M+H]+"
    assert qs.polarity == "positive"
    assert qs.collision_energy == 20.0
    assert qs.mz_array == [1.0]
    assert qs.intensity_array == [10.0]

def test_spectrumdata_auto_conversion():
    sd = SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=None, mz_array=[1.0], intensity_array=[10.0])
    assert isinstance(sd.mz_array, np.ndarray)
    assert isinstance(sd.intensity_array, np.ndarray)

# SpectrumLoader
def test_spectrumloader_normalize_aliases():
    row = {"PrecursorMZ": 123.4, "IonMode": "-", "peaks_mz": [1], "exact_mass": 100.0}
    norm = SpectrumLoader._normalize_row_columns(row)
    assert norm["precursor_mz"] == 123.4
    assert norm["polarity"] == "-"
    assert norm["mz_array"] == [1]
    assert norm["neutral_mass"] == 100.0

def test_spectrumloader_polarity_normalization():
    sd1 = SpectrumLoader.from_row_dict({"precursor_mz": 100.0, "polarity": "+"})
    assert sd1.polarity == "positive"
    
    sd2 = SpectrumLoader.from_row_dict({"precursor_mz": 100.0, "polarity": "-"})
    assert sd2.polarity == "negative"
    
    sd3 = SpectrumLoader.from_row_dict({"precursor_mz": 100.0, "polarity": "1"})
    assert sd3.polarity == "positive"

def test_spectrumloader_polarity_from_adduct():
    sd1 = SpectrumLoader.from_row_dict({"precursor_mz": 100.0, "adduct": "[M-H]-", "polarity": "unknown"})
    assert sd1.polarity == "negative"

def test_spectrumloader_load_file_unsupported():
    with pytest.raises(ValueError, match="Unsupported file format"):
        SpectrumLoader.load_file("test.json")

def test_spectrumloader_load_csv_not_found():
    with pytest.raises(FileNotFoundError):
        SpectrumLoader.load_csv("nonexistent.csv")

def test_spectrumloader_load_parquet_not_found():
    with pytest.raises(FileNotFoundError):
        SpectrumLoader.load_parquet("nonexistent.parquet")

def test_spectrumloader_to_dataframe_roundtrip():
    sd = SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=20.0, mz_array=[1.0], intensity_array=[10.0])
    df = SpectrumLoader.to_dataframe([sd])
    assert len(df) == 1
    assert df.iloc[0]["molecule_id"] == "1"
    assert df.iloc[0]["mz_array"] == [1.0]

def test_spectrumloader_save_load_csv_roundtrip(tmp_path):
    sd = SpectrumData(molecule_id="1", precursor_mz=100.0, adduct="[M+H]+", polarity="positive", collision_energy=20.0, mz_array=[1.0], intensity_array=[10.0], inchikey14="ABCDEFGHIJKLMN")
    p = tmp_path / "test.csv"
    SpectrumLoader.save_csv([sd], p)
    loaded = SpectrumLoader.load_csv(p)
    assert len(loaded) == 1
    assert loaded[0].molecule_id == "1"
    np.testing.assert_allclose(loaded[0].mz_array, [1.0])
    np.testing.assert_allclose(loaded[0].intensity_array, [100.0])

# MockDataGenerator
def test_mock_generate_spectrum():
    gen = MockDataGenerator()
    spec = gen.generate_spectrum()
    assert isinstance(spec, SpectrumData)
    assert spec.precursor_mz > 0

def test_mock_generate_dataset():
    gen = MockDataGenerator()
    ds = gen.generate_dataset(n_samples=5)
    assert len(ds) == 5
    for spec in ds:
        assert isinstance(spec, SpectrumData)

def test_mock_generate_mock_library():
    gen = MockDataGenerator()
    df, emb = gen.generate_mock_library(n_entries=10, embedding_dim=16)
    assert len(df) == 10
    assert emb.shape == (10, 16)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def test_mock_generate_mock_candidate_db():
    gen = MockDataGenerator()
    db = gen.generate_mock_candidate_db(n_candidates=10, n_bits=256)
    assert len(db["candidate_df"]) == 10
    assert db["fingerprints"].shape == (10, 4)

def test_mock_generate_dataset_no_mispick():
    gen = MockDataGenerator()
    ds = gen.generate_dataset(n_samples=5, mispick_ratio=0.0)
    for spec in ds:
        assert not spec.is_13c_mispick
