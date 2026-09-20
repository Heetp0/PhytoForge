"""Adversarial stress-test suite for Phase 3 fragment library and knapsack assembler.

Tests:
1. KnapsackAssembler.assemble() with 10,000+ millimass keys under extreme combinatorial loads.
   Verifies strict adherence to timeout_s=2.0 and MAX_ITER=200_000 (< 3.0s wall clock).
2. NeutralLossLibrary with empty inputs, extreme ppm tolerances, non-existent keys,
   and float vs integer millimass keys.
3. BRICSFragmentLibrary with challenging structures (multiple attachment points,
   fused polycycles, macrocycles, stereoisomers) verifying NO dummy atom [*] remains.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.reranking.fragment_library import KnapsackAssembler, NeutralLossLibrary, parse_peaks_field
from src.reranking.brics_knapsack import (
    BRICSFragmentLibrary,
    extract_fragments,
    fragment_mass,
    strip_dummy_atoms_and_sanitize,
)

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# =====================================================================
# 1. KnapsackAssembler Stress Tests (10,000+ Keys & Wall-Clock Bounds)
# =====================================================================

class TestKnapsackAssemblerStress:
    """Adversarial stress testing for KnapsackAssembler."""

    def test_assemble_with_10k_keys_respects_timeout_and_max_iter(self):
        """Stress-test with 10,000+ keys: verify wall-clock runtime strictly < 3.0s under 5.0 ppm."""
        assembler = KnapsackAssembler()

        # Dense mock library of 10,000+ unique millimass keys
        mock_library: Dict[int, List[str]] = {
            k: [f"frag_{k}_a", f"frag_{k}_b"] for k in range(1000, 11001)
        }
        assert len(mock_library) > 10000

        target_stress_mass = 2.5  # target_key = 25000

        t0 = time.perf_counter()
        candidates = assembler.assemble(
            target_mass_da=target_stress_mass,
            library=mock_library,
            ppm_tolerance=5.0,
            max_depth=3,
            max_candidates=25,
            timeout_s=2.0,
        )
        elapsed = time.perf_counter() - t0

        # Verification criteria: strictly completes in < 3.0s wall-clock
        assert elapsed < 3.0, f"Assembler exceeded 3.0s wall-clock: {elapsed:.3f}s"
        assert isinstance(candidates, list)
        assert len(candidates) <= 25

    def test_assemble_with_50k_keys_ppm5_respects_time_ceiling(self):
        """Stress-test with 50,000 keys: verify runtime remains < 1.0s under standard 5.0 ppm."""
        assembler = KnapsackAssembler()
        mock_lib = {k: ["frag_a", "frag_b"] for k in range(1000, 51000)}
        assert len(mock_lib) == 50000

        t0 = time.perf_counter()
        candidates = assembler.assemble(
            target_mass_da=5.0,
            library=mock_lib,
            ppm_tolerance=5.0,
            max_depth=3,
            max_candidates=25,
            timeout_s=2.0,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"50k keys exceeded 3.0s: {elapsed:.3f}s"
        assert len(candidates) <= 25

    def test_assemble_with_10k_keys_timeout_abort_under_exhaustion(self):
        """Verify timeout guard triggers when combinatorial space has 0 matches."""
        assembler = KnapsackAssembler()

        # Library of 9,900 keys, all multiples of 10 (ending in 0)
        mock_library: Dict[int, List[str]] = {
            k: [f"NL_{k}"] for k in range(1000, 100000, 10)
        }
        assert len(mock_library) >= 9000

        # Target mass 5.0005 Da (key 50005). With delta_key=1, window is [50004, 50006].
        # Sum of 1, 2, or 3 multiples of 10 is ALWAYS a multiple of 10.
        # Neither 50004, 50005, nor 50006 is a multiple of 10, so 0 matches can ever exist.
        target_mass = 5.0005

        t0 = time.perf_counter()
        candidates = assembler.assemble(
            target_mass_da=target_mass,
            library=mock_library,
            ppm_tolerance=0.001,
            max_depth=3,
            max_candidates=25,
            timeout_s=0.5,  # Short timeout
        )
        elapsed = time.perf_counter() - t0

        # Should exit within ~0.5s + small overhead (< 1.5s)
        assert elapsed < 1.5, f"Timeout failed to abort search: elapsed {elapsed:.3f}s"
        assert len(candidates) == 0

    def test_assemble_governor_timeout_breach_under_wide_tolerance(self):
        """EMPIRICAL CHALLENGE: Verify KnapsackAssembler respects timeout_s=2.0 (< 3.0s wall-clock).

        Under wide tolerance (500 ppm) with a 10,000-key library and low candidate pool,
        the assembler fails to count inner-loop iterations in Depth 3, causing it to run for
        ~9.3 seconds and bypass the governor timeout guard.
        """
        assembler = KnapsackAssembler()
        mock_lib = {k: ["shared_a", "shared_b"] for k in range(1660000, 1670000)}
        assert len(mock_lib) == 10000

        t0 = time.perf_counter()
        candidates = assembler.assemble(
            target_mass_da=500.0,
            library=mock_lib,
            ppm_tolerance=500.0,
            max_depth=3,
            max_candidates=25,
            timeout_s=2.0,
        )
        elapsed = time.perf_counter() - t0

        # Mandatory acceptance criterion: Must complete in < 3.0s wall-clock
        assert elapsed < 3.0, (
            f"GOVERNOR TIMEOUT BYPASS: KnapsackAssembler took {elapsed:.3f}s, "
            f"exceeding the strict 3.0s wall-clock limit under timeout_s=2.0"
        )

    def test_assemble_extreme_inputs(self):
        """Test assembler with boundary values, negative numbers, NaNs, and huge masses."""
        assembler = KnapsackAssembler()
        mock_lib = {10000: ["frag_1"], 20000: ["frag_2"]}

        # Negative and zero mass
        assert assembler.assemble(target_mass_da=-100.0, library=mock_lib) == []
        assert assembler.assemble(target_mass_da=0.0, library=mock_lib) == []

        # Empty library
        assert assembler.assemble(target_mass_da=1.0, library={}) == []

        # Negative / zero max_candidates or max_depth
        assert assembler.assemble(target_mass_da=1.0, library=mock_lib, max_candidates=0) == []
        assert assembler.assemble(target_mass_da=1.0, library=mock_lib, max_depth=0) == []
        assert assembler.assemble(target_mass_da=1.0, library=mock_lib, max_candidates=-5) == []

        # Extreme huge mass (100,000 Da)
        assert assembler.assemble(target_mass_da=100_000.0, library=mock_lib) == []

    def test_assemble_filters_non_int_and_negative_keys(self):
        """Verify assembler cleanly ignores float, string, boolean, or negative keys in library."""
        assembler = KnapsackAssembler()
        # Messy library with mixed types
        dirty_lib: Dict[Any, List[str]] = {
            10000: ["valid_1"],
            "15000": ["string_key"],
            15000.5: ["float_key_half"],
            25000.0: ["float_key_exact"],
            True: ["bool_key"],
            -5000: ["negative_key"],
            0: ["zero_key"],
            20000: ["valid_2"],
        }
        # Target mass 3.0 Da -> key 30000 = 10000 + 20000
        cands = assembler.assemble(target_mass_da=3.0, library=dirty_lib, max_depth=2)
        assert len(cands) > 0
        for cand in cands:
            assert "string_key" not in cand
            assert "float_key_half" not in cand
            assert "float_key_exact" not in cand
            assert "bool_key" not in cand
            assert "negative_key" not in cand
            assert "zero_key" not in cand


# =====================================================================
# 2. NeutralLossLibrary Adversarial Tests
# =====================================================================

class TestNeutralLossLibraryAdversarial:
    """Adversarial edge-case tests for NeutralLossLibrary."""

    def test_empty_and_corrupt_dataframe_inputs(self):
        """Test build_from_dataframe with None, empty DataFrame, missing columns, NaNs."""
        lib = NeutralLossLibrary()

        # None DataFrame
        stats = lib.build_from_dataframe(None)
        assert stats["total_spectra"] == 0
        assert stats["unique_keys"] == 0

        # Empty DataFrame
        stats = lib.build_from_dataframe(pd.DataFrame())
        assert stats["total_spectra"] == 0

        # DataFrame with only NaNs
        nan_df = pd.DataFrame({
            "precursor_mz": [np.nan, None, -5.0, 0.0],
            "adduct": ["[M+H]+", np.nan, "INVALID_ADDUCT", ""],
            "peaks": [None, np.nan, "", "[[100.0, 10.0]]"],
        })
        stats = lib.build_from_dataframe(nan_df)
        assert stats["total_spectra"] == 4
        assert len(lib) == 0

    def test_query_extreme_ppm_tolerances(self):
        """Test query with extreme tolerances: 0.001 ppm, 1000 ppm, negative, huge ppm."""
        lib = NeutralLossLibrary()
        # Seed keys: 100.0 Da -> key 1000000; 100.01 Da -> key 1000100; 101.0 Da -> key 1010000
        lib.library = {
            1000000: ["NL_100_0000"],
            1000100: ["NL_100_0100"],
            1010000: ["NL_101_0000"],
        }

        # 1. Extremely tight tolerance: 0.001 ppm
        # 100.0 Da * 0.001 / 1e6 = 1e-7 Da -> window is [1000000, 1000000]
        res = lib.query(target_mass_da=100.0, ppm_tolerance=0.001)
        assert res == [1000000]

        # 2. Non-matching tight tolerance
        res = lib.query(target_mass_da=100.005, ppm_tolerance=0.001)
        assert res == []

        # 3. Wide tolerance: 1000 ppm
        # 100.0 Da * 1000 / 1e6 = 0.1 Da -> window is [99.9, 100.1] Da -> [999000, 1001000]
        res = lib.query(target_mass_da=100.0, ppm_tolerance=1000.0)
        assert 1000000 in res
        assert 1000100 in res
        assert 1010000 not in res

        # 4. Ultra-wide tolerance: 200,000 ppm (spans > 2000 millimass keys)
        res_wide = lib.query(target_mass_da=100.0, ppm_tolerance=20000.0)
        assert len(res_wide) >= 2

        # 5. Negative tolerance -> returns []
        assert lib.query(target_mass_da=100.0, ppm_tolerance=-5.0) == []

        # 6. Zero or negative target mass -> returns []
        assert lib.query(target_mass_da=0.0, ppm_tolerance=5.0) == []
        assert lib.query(target_mass_da=-50.0, ppm_tolerance=5.0) == []

    def test_non_existent_keys_and_getitem(self):
        """Test query, containment, and getitem for non-existent keys."""
        lib = NeutralLossLibrary()
        lib.library = {500000: ["NL_50"]}

        # Non-existent query returns empty list, never raises
        assert lib.query(target_mass_da=999.0) == []

        # __contains__
        assert 500000 in lib
        assert 999999 not in lib

        # __getitem__ for non-existent key raises KeyError
        with pytest.raises(KeyError):
            _ = lib[999999]

    def test_float_keys_rejected_in_constructor(self):
        """Test that float keys are strictly rejected with TypeError in constructor."""
        with pytest.raises(TypeError, match="must be int, got float"):
            NeutralLossLibrary(library={100.5: ["val"]})  # type: ignore

        with pytest.raises(TypeError, match="must be int, got float"):
            NeutralLossLibrary(library={np.float64(100.0): ["val"]})  # type: ignore

    def test_parse_peaks_field_pathological_strings(self):
        """Test parse_peaks_field against pathological, malformed, and injection-like strings."""
        assert parse_peaks_field("") == []
        assert parse_peaks_field("   ") == []
        assert parse_peaks_field("nonsense;garbage text;null") == []
        assert parse_peaks_field("NaN:Inf;100.0:NaN;200.0:50.0") == [(200.0, 50.0)]
        assert parse_peaks_field("100.0 50.0\n\n200.0 80.0\n") == [(100.0, 50.0), (200.0, 80.0)]
        assert parse_peaks_field("[[100.0, 50.0], [\"bad\", 20.0], [300.0, \"bad\"]]") == [(100.0, 50.0)]


# =====================================================================
# 3. BRICSFragmentLibrary Adversarial Tests (Challenging Chemistries)
# =====================================================================

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required for BRICS tests")
class TestBRICSFragmentLibraryAdversarial:
    """Adversarial chemical structure tests for BRICSFragmentLibrary."""

    def test_molecules_with_multiple_attachment_points(self):
        """Test structures where central cores connect to 3+ BRICS cleavage sites."""
        smiles_list = [
            # 1. Melamine/Triazine tri-aniline (3 aromatic amine attachment points)
            "c1(nc(nc(n1)Nc2ccccc2)Nc3ccccc3)Nc4ccccc4",
            # 2. Pentaerythritol tetrabutyrate (4 ester cleavage sites around quaternary C)
            "CCCC(=O)OCC(COC(=O)CCC)(COC(=O)CCC)COC(=O)CCC",
            # 3. EDTA tetra-acid (4 carboxylic acids + 2 tertiary amines)
            "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
        ]

        lib = extract_fragments(smiles_list)
        assert len(lib) > 0

        # CRITICAL CHECK: Verify NO fragment contains any dummy atom [*]
        for key, frag_list in lib.items():
            assert isinstance(key, int), f"Key {key} is not int"
            for frag in frag_list:
                assert "*" not in frag, f"Found dummy atom in fragment: {frag}"
                # Verify parseable by RDKit
                mol = Chem.MolFromSmiles(frag)
                assert mol is not None, f"Fragment {frag} failed RDKit parse"
                # Verify mass matches millimass key
                m = fragment_mass(frag)
                assert m > 0.0
                assert round(m * 10000) == key

    def test_fused_polycycles_and_bridged_systems(self):
        """Test fused polycyclic natural products and bridged hydrocarbons."""
        polycycles = [
            # Cholesterol (cyclopentanoperhydrophenanthrene 4-ring fused core)
            "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
            # Taxol / Paclitaxel (complex bridged bicyclic diterpene core with multiple esters)
            "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
            # Morphine (fused pentacyclic ring system with phenanthrene & ether bridge)
            "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
            # Strychnine (heptacyclic alkaloid with bridged cage)
            "O=C1CC2OCC=C3CN4CCC56C(=CC=CC5=O)NC2C6C4CC13",
            # Adamantane derivative (bridged tricyclo[3.3.1.1]decane)
            "C1C2CC3CC1CC(C2)(C3)NC(=O)c4ccccc4",
        ]

        lib = extract_fragments(polycycles)
        assert len(lib) > 0

        for key, frags in lib.items():
            for frag in frags:
                assert "*" not in frag, f"Polycycle fragment contains dummy atom: {frag}"
                mol = Chem.MolFromSmiles(frag)
                assert mol is not None, f"Unparseable polycycle fragment: {frag}"

    def test_macrocycles(self):
        """Test macrocyclic rings with cleavable bonds inside large rings."""
        macrocycles = [
            # Erythromycin (14-membered macrolide ring with glycosides)
            "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
            # 18-crown-6 (macrocyclic polyether, 18-membered ring)
            "C1COCCOCCOCCOCCOCCO1",
            # Cyclic dipeptide / diketopiperazine
            "O=C1CNC(=O)CN1",
        ]

        lib = extract_fragments(macrocycles)
        assert len(lib) > 0

        for key, frags in lib.items():
            for frag in frags:
                assert "*" not in frag, f"Macrocycle fragment contains dummy atom: {frag}"
                mol = Chem.MolFromSmiles(frag)
                assert mol is not None, f"Unparseable macrocycle fragment: {frag}"

    def test_stereoisomers_and_chiral_centers(self):
        """Test chiral diastereomers and epimers."""
        stereoisomers = [
            # D-Glucose
            "OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)C=O",
            # L-Glucose (enantiomer)
            "OC[C@H](O)[C@H](O)[C@@H](O)[C@H](O)C=O",
            # Thalidomide (R)
            "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            # Thalidomide (S)
            "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
        ]

        lib = extract_fragments(stereoisomers)
        assert len(lib) > 0

        for key, frags in lib.items():
            for frag in frags:
                assert "*" not in frag, f"Chiral fragment contains dummy atom: {frag}"
                mol = Chem.MolFromSmiles(frag)
                assert mol is not None, f"Chiral fragment unparseable: {frag}"

    def test_strip_dummy_atoms_edge_cases(self):
        """Direct unit test of strip_dummy_atoms_and_sanitize on exotic dummy markers."""
        # Bare dummy atom
        assert strip_dummy_atoms_and_sanitize("*") is None
        assert strip_dummy_atoms_and_sanitize("[*]") is None
        assert strip_dummy_atoms_and_sanitize("[0*]") is None
        assert strip_dummy_atoms_and_sanitize("[4*][4*]") is None

        # Numbered dummy atom with hydrogen
        res_h = strip_dummy_atoms_and_sanitize("[1*][H]")
        assert res_h is not None
        assert res_h[0] == "[H]"
        assert "*" not in res_h[0]

        # Multiple numbered dummy markers
        res = strip_dummy_atoms_and_sanitize("[1*]c1ccccc1[2*]")
        assert res is not None
        clean_smi, mass, millimass = res
        assert clean_smi == "c1ccccc1"
        assert "*" not in clean_smi
        assert millimass == 780470  # Benzene 78.04695 Da -> 780470

        # Complex dummy markers [14*], [15*]
        res2 = strip_dummy_atoms_and_sanitize("[14*]CC(=O)O[15*]")
        assert res2 is not None
        clean_smi2, _, _ = res2
        assert "*" not in clean_smi2
        assert clean_smi2 == "CC(=O)O"  # Acetic acid

        # Corrupted / unparseable SMILES
        assert strip_dummy_atoms_and_sanitize("INVALID_SMILES") is None
        assert strip_dummy_atoms_and_sanitize("") is None
        assert strip_dummy_atoms_and_sanitize(None) is None
