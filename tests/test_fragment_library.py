"""
Unit tests for NeutralLossLibrary and KnapsackAssembler in src/reranking/fragment_library.py.

Verifies:
- NeutralLossLibrary:
  - build_from_dataframe with realistic spectra and known adducts
  - save/load roundtrip (JSON and Pickle)
  - int key verification (strictly int keys, no float keys)
  - query with ppm tolerance
  - empty corpus handling
  - malformed and corrupt peak strings / missing fields resilience
- KnapsackAssembler:
  - empty library handling
  - exact depth-1 match
  - depth-2 combinations match
  - depth-3 combinations match
  - timeout guard execution within 3 seconds on large library (5000+ keys)
  - max_candidates enforcement
  - candidate deduplication
  - integer key arithmetic
- parse_peaks_field:
  - multiple representation formats (JSON, delimited, 2D array, separate columns)
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.chemistry.adducts import calculate_neutral_mass
from src.reranking.fragment_library import (
    KnapsackAssembler,
    NeutralLossLibrary,
    parse_peaks_field,
)


# ============================================================================
# Fixtures and Helpers
# ============================================================================

@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """
    Constructs a synthetic training DataFrame with known precursor m/z, adducts, and peaks.
    Molecule 1: Caffeine neutral mass ~194.080376 Da
      Adduct: [M+H]+ -> precursor_mz = 195.087652
      Neutral loss 1: water loss (18.0106 Da) -> fragment mz = 194.080376 - 18.0106 = 176.0698
      Neutral loss 2: CO loss (27.9949 Da) -> fragment mz = 194.080376 - 27.9949 = 166.0855
    Molecule 2: Resveratrol neutral mass ~228.078644 Da
      Adduct: [M-H]- -> precursor_mz = 227.071368
      Neutral loss 1: C2H2O loss (42.0106 Da) -> fragment mz = 228.078644 - 42.0106 = 186.0680
    """
    rows = [
        {
            "molecule_id": "MOL_CAFFEINE",
            "precursor_mz": 195.087652,
            "adduct": "[M+H]+",
            "peaks": "176.0698:100.0;166.0855:50.0;50.0:0.001",  # 50.0 is below 0.01 relative intensity
            "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        },
        {
            "molecule_id": "MOL_RESVERATROL",
            "precursor_mz": 227.071368,
            "adduct": "[M-H]-",
            "peaks": "[[186.0680, 80.0], [100.0, 0.0001]]",
            "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        },
    ]
    return pd.DataFrame(rows)


# ============================================================================
# Tests for parse_peaks_field
# ============================================================================

def test_parse_peaks_field_formats():
    # Semicolon and colon
    p1 = parse_peaks_field("105.03:10.5;121.02:45.2")
    assert len(p1) == 2
    assert pytest.approx(p1[0][0]) == 105.03
    assert pytest.approx(p1[0][1]) == 10.5

    # Space separated with semicolon
    p2 = parse_peaks_field("105.03 10.5;121.02 45.2")
    assert len(p2) == 2
    assert pytest.approx(p2[1][0]) == 121.02
    assert pytest.approx(p2[1][1]) == 45.2

    # JSON list of pairs
    p3 = parse_peaks_field("[[105.03, 10.5], [121.02, 45.2]]")
    assert len(p3) == 2
    assert pytest.approx(p3[0][0]) == 105.03

    # 2D numpy array
    arr = np.array([[105.03, 10.5], [121.02, 45.2]])
    p4 = parse_peaks_field(arr)
    assert len(p4) == 2
    assert pytest.approx(p4[0][0]) == 105.03

    # 1D numpy array
    arr_1d = np.array([105.03, 121.02])
    p5 = parse_peaks_field(arr_1d)
    assert len(p5) == 2
    assert pytest.approx(p5[0][1]) == 100.0


def test_parse_peaks_field_resilience():
    # Malformed strings
    assert parse_peaks_field("invalid;malformed:text") == []
    assert parse_peaks_field("") == []
    assert parse_peaks_field(None) == []
    assert parse_peaks_field(float("nan")) == []

    # Row with separate mz_array and intensity_array
    row = {
        "mz_array": [105.03, 121.02],
        "intensity_array": [10.5, 45.2],
    }
    p_row = parse_peaks_field(None, row=row)
    assert len(p_row) == 2
    assert pytest.approx(p_row[0][0]) == 105.03
    assert pytest.approx(p_row[0][1]) == 10.5


# ============================================================================
# Tests for NeutralLossLibrary
# ============================================================================

def test_neutral_loss_library_build_from_dataframe(sample_training_df):
    lib = NeutralLossLibrary(min_intensity=0.01, max_mass=500.0)
    stats = lib.build_from_dataframe(sample_training_df)

    assert stats["total_spectra"] == 2
    assert stats["unique_keys"] >= 2
    assert len(lib) == stats["unique_keys"]

    # Verify keys are integers and within expected ranges
    for k, v in lib.library.items():
        assert isinstance(k, int)
        assert not isinstance(k, bool)
        assert not isinstance(k, float)
        assert isinstance(v, list)
        assert len(v) > 0

    # Check water loss (~18.0106 Da -> millimass ~180106)
    expected_water_key = int(round(18.0106 * 10000))
    # Query with 50 ppm window around water
    matches = lib.query(18.0106, ppm_tolerance=50.0)
    assert len(matches) > 0
    assert any(abs(m - expected_water_key) <= 5 for m in matches)


def test_neutral_loss_library_empty_corpus():
    lib = NeutralLossLibrary()
    # Empty DataFrame with no columns
    stats1 = lib.build_from_dataframe(pd.DataFrame())
    assert stats1["total_spectra"] == 0
    assert stats1["unique_keys"] == 0
    assert len(lib) == 0

    # Empty DataFrame with expected columns
    df_empty = pd.DataFrame(columns=["precursor_mz", "adduct", "peaks", "smiles"])
    stats2 = lib.build_from_dataframe(df_empty)
    assert stats2["total_spectra"] == 0
    assert stats2["unique_keys"] == 0


def test_neutral_loss_library_malformed_rows():
    lib = NeutralLossLibrary()
    df_bad = pd.DataFrame([
        {"precursor_mz": -100.0, "adduct": "[M+H]+", "peaks": "100.0:50.0"},  # Negative precursor mz
        {"precursor_mz": 200.0, "adduct": "UNKNOWN_ADDUCT", "peaks": "100.0:50.0"},  # Invalid adduct
        {"precursor_mz": 200.0, "adduct": "[M+H]+", "peaks": "garbage:data;foo"},  # Corrupted peaks
        {"precursor_mz": 200.0, "adduct": "[M+H]+", "peaks": None},  # None peaks
        {"precursor_mz": None, "adduct": "[M+H]+", "peaks": "100.0:50.0"},  # None mz
        {"precursor_mz": 200.0, "adduct": None, "peaks": "100.0:50.0"},  # None adduct
        {  # One valid row
            "precursor_mz": 200.0,
            "adduct": "[M+H]+",
            "peaks": "150.0:100.0",
            "smiles": "CC(=O)O",
        },
    ])

    stats = lib.build_from_dataframe(df_bad)
    assert stats["total_spectra"] == 7
    # Exactly one valid fragment should be extracted
    assert stats["unique_keys"] == 1
    assert len(lib) == 1


def test_neutral_loss_library_int_keys_strictly_enforced():
    # Passing float keys directly to constructor must raise TypeError
    with pytest.raises(TypeError, match="must be int"):
        NeutralLossLibrary(library={100.5: ["invalid"]})

    # Verify no float keys exist in generated library
    lib = NeutralLossLibrary(library={100: ["fragA"], 200: ["fragB"]})
    for k in lib.library.keys():
        assert type(k) is int
        assert not isinstance(k, float)


def test_neutral_loss_library_save_load_pickle(tmp_path: Path):
    lib = NeutralLossLibrary(library={180106: ["H2O"], 279949: ["CO"]})
    pkl_path = tmp_path / "fragments.pkl"

    lib.save(pkl_path)
    assert pkl_path.exists()

    loaded = NeutralLossLibrary.from_file(pkl_path)
    assert len(loaded) == 2
    assert loaded[180106] == ["H2O"]
    assert loaded[279949] == ["CO"]

    # Verify all loaded keys are strictly int
    for k in loaded.library.keys():
        assert type(k) is int


def test_neutral_loss_library_save_load_json(tmp_path: Path):
    lib = NeutralLossLibrary(library={180106: ["H2O"], 279949: ["CO"]})
    json_path = tmp_path / "fragments.json"

    lib.save(json_path)
    assert json_path.exists()

    loaded = NeutralLossLibrary()
    loaded.load(json_path)
    assert len(loaded) == 2
    assert loaded[180106] == ["H2O"]
    assert loaded[279949] == ["CO"]

    # Ensure JSON string keys were converted back to int
    for k in loaded.library.keys():
        assert type(k) is int


def test_neutral_loss_library_query_ppm_tolerance():
    # Library with keys:
    # 1000000 -> 100.0000 Da
    # 1000005 -> 100.0005 Da (+5.0 ppm)
    # 1000020 -> 100.0020 Da (+20.0 ppm)
    # 2000000 -> 200.0000 Da
    lib = NeutralLossLibrary(library={
        1000000: ["frag_exact"],
        1000005: ["frag_plus_5ppm"],
        1000020: ["frag_plus_20ppm"],
        2000000: ["frag_200"],
    })

    # At 5.0 ppm on 100.0 Da, window is ±0.0005 Da (±5 millimass units)
    res_5ppm = lib.query(100.0, ppm_tolerance=5.0)
    assert 1000000 in res_5ppm
    assert 1000005 in res_5ppm
    assert 1000020 not in res_5ppm
    assert 2000000 not in res_5ppm

    # At 25.0 ppm, 1000020 should now match
    res_25ppm = lib.query(100.0, ppm_tolerance=25.0)
    assert 1000020 in res_25ppm

    # Edge cases
    assert lib.query(-10.0) == []
    assert lib.query(0.0) == []
    assert NeutralLossLibrary().query(100.0) == []


# ============================================================================
# Tests for KnapsackAssembler
# ============================================================================

def test_knapsack_assembler_empty_library():
    assembler = KnapsackAssembler()
    assert assembler.assemble(100.0, {}) == []
    assert assembler.assemble(-100.0, {1000000: ["A"]}) == []
    assert assembler.assemble(100.0, {1000000: ["A"]}, max_candidates=0) == []
    assert assembler.assemble(100.0, {1000000: ["A"]}, max_depth=0) == []


def test_knapsack_assembler_exact_match():
    assembler = KnapsackAssembler()
    library = {
        1000000: ["frag100_exact"],
        2000000: ["frag200"],
    }
    # Query 100.0 Da
    candidates = assembler.assemble(100.0, library, ppm_tolerance=5.0, max_depth=1)
    assert len(candidates) == 1
    assert candidates[0] == "frag100_exact"


def test_knapsack_assembler_combinations_match():
    assembler = KnapsackAssembler()
    # 40 Da (400000) + 60 Da (600000) = 100 Da (1000000)
    # 20 Da (200000) + 30 Da (300000) + 50 Da (500000) = 100 Da (1000000)
    library = {
        200000: ["f20"],
        300000: ["f30"],
        400000: ["f40"],
        500000: ["f50"],
        600000: ["f60"],
    }

    # Depth 2 combination match
    res_d2 = assembler.assemble(100.0, library, ppm_tolerance=5.0, max_depth=2)
    assert any("f40+f60" in c for c in res_d2)

    # Depth 3 combination match
    res_d3 = assembler.assemble(100.0, library, ppm_tolerance=5.0, max_depth=3)
    assert any("f20+f30+f50" in c for c in res_d3)


def test_knapsack_assembler_candidate_deduplication():
    assembler = KnapsackAssembler()
    library = {
        400000: ["fragA", "fragA"],  # Duplicate entry
        600000: ["fragB"],
    }
    candidates = assembler.assemble(100.0, library, ppm_tolerance=5.0, max_depth=2)
    # Deduplication check
    assert len(candidates) == len(set(candidates))


def test_knapsack_assembler_max_candidates_respected():
    assembler = KnapsackAssembler()
    # Create library with many combinations
    library = {i * 10000: [f"frag_{i}"] for i in range(1, 100)}
    candidates = assembler.assemble(
        100.0,
        library,
        ppm_tolerance=100.0,
        max_depth=3,
        max_candidates=7,
    )
    assert len(candidates) == 7


def test_knapsack_assembler_timeout_guard_on_large_library():
    """
    Verifies that KnapsackAssembler returns well within 3 seconds when executing
    against a large library with 5000+ keys.
    """
    assembler = KnapsackAssembler()
    # Generate 6000 keys between 10 Da and 500 Da
    large_library = {
        int(k): [f"NL_{k}"]
        for k in np.linspace(100000, 5000000, 6000, dtype=int)
    }

    t0 = time.perf_counter()
    # Target 300.0 Da with 2.0s timeout
    candidates = assembler.assemble(
        target_mass_da=300.0,
        library=large_library,
        ppm_tolerance=20.0,
        max_depth=3,
        max_candidates=25,
        timeout_s=2.0,
    )
    elapsed = time.perf_counter() - t0

    # Strict requirement: returns within 3.0 seconds
    assert elapsed < 3.0, f"Assembler exceeded 3s timeout guard: elapsed={elapsed:.3f}s"
    assert isinstance(candidates, list)


def test_knapsack_assembler_integer_key_arithmetic():
    assembler = KnapsackAssembler()
    # Ensure keys that are float-like in library are handled safely
    library: Dict[int, List[str]] = {
        1000000: ["cand100"],
    }
    # assemble should perform all searches with integer arithmetic
    res = assembler.assemble(100.0, library)
    assert res == ["cand100"]


def test_neutral_loss_library_build_from_file_csv_and_parquet(tmp_path: Path, sample_training_df: pd.DataFrame):
    csv_path = tmp_path / "train.csv"
    parquet_path = tmp_path / "train.parquet"

    sample_training_df.to_csv(csv_path, index=False)
    sample_training_df.to_parquet(parquet_path, index=False)

    lib_csv = NeutralLossLibrary(min_intensity=0.01, max_mass=500.0)
    stats_csv = lib_csv.build_from_file(csv_path)
    assert stats_csv["total_spectra"] == 2
    assert stats_csv["unique_keys"] >= 2

    lib_parquet = NeutralLossLibrary(min_intensity=0.01, max_mass=500.0)
    stats_parquet = lib_parquet.build_from_file(parquet_path)
    assert stats_parquet["total_spectra"] == 2
    assert stats_parquet["unique_keys"] == stats_csv["unique_keys"]

    # Verify identical keys
    assert set(lib_csv.library.keys()) == set(lib_parquet.library.keys())


def test_neutral_loss_library_intensity_and_mass_filtering():
    # Molecule with precursor mz = 200.0, adduct = [M+H]+ -> neutral mass = 198.992724
    # Peak 1: mz = 190.0 -> neutral loss = 8.992724, intensity = 100.0 (base peak)
    # Peak 2: mz = 180.0 -> neutral loss = 18.992724, intensity = 0.5 (0.005 rel intensity < 0.01)
    # Peak 3: mz = 10.0 -> neutral loss = 188.992724, intensity = 50.0 (exceeds max_mass=100.0)
    # Peak 4: mz = 210.0 -> neutral loss = -11.007276 (unphysical negative loss)
    df = pd.DataFrame([{
        "precursor_mz": 200.0,
        "adduct": "[M+H]+",
        "peaks": "190.0:100.0;180.0:0.5;10.0:50.0;210.0:50.0",
        "smiles": "TEST_MOL",
    }])

    lib = NeutralLossLibrary(min_intensity=0.01, max_mass=100.0)
    stats = lib.build_from_dataframe(df)

    # Only Peak 1 should survive!
    assert stats["unique_keys"] == 1
    surviving_key = list(lib.library.keys())[0]
    expected_key = int(round((198.992723548 - 190.0) * 10000))
    assert surviving_key == expected_key


def test_knapsack_assembler_depth3_combination_with_replacement():
    assembler = KnapsackAssembler()
    # Target 90.0 Da -> 900000 millimass
    # Key 300000 used 3 times: 300000 + 300000 + 300000 = 900000
    library = {
        300000: ["frag30"],
    }
    res = assembler.assemble(90.0, library, ppm_tolerance=5.0, max_depth=3)
    assert len(res) == 1
    assert res[0] == "frag30+frag30+frag30"


def test_knapsack_assembler_no_match():
    assembler = KnapsackAssembler()
    library = {
        100000: ["frag10"],
        200000: ["frag20"],
    }
    # Target 500.0 Da cannot be formed with depth 3 and max fragment 20 Da (max sum = 60 Da)
    res = assembler.assemble(500.0, library, ppm_tolerance=5.0, max_depth=3)
    assert res == []

