"""
tests/test_botanical_knowledge.py
Comprehensive unit and integration test suite for the expanded 16-transformation
botanical knowledge base, bidirectional delta matching, tolerance limits, and
MS2 fragment co-validation in transductive molecular networking (Milestone M5, R3).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.transductive_networking import (
    BOTANICAL_TRANSFORMATIONS,
    DELTA_LIBRARY,
    NetworkCandidate,
    TransductiveMolecularNetwork,
)


# ---------------------------------------------------------------------------
# All 16 Verified Natural Product Transformations Definition
# ---------------------------------------------------------------------------

VERIFIED_16_TRANSFORMATIONS = [
    # Glycosylations
    ("Hexose", 162.0528),
    ("Pentose", 132.0423),
    ("Rhamnose", 146.0579),
    ("Glucuronide", 176.0321),
    # Acylations & Alkylations
    ("Methyl", 14.0157),
    ("Acetyl", 42.0106),
    ("Malonyl", 86.0004),
    ("Prenyl", 68.0626),
    # Phenylpropanoids
    ("Galloyl", 152.0110),
    ("Caffeoyl", 162.0317),
    ("Feruloyl", 176.0473),
    ("Coumaroyl", 146.0368),
    ("Sinapoyl", 206.0579),
    # Functionalizations
    ("Hydroxylation", 15.9949),
    ("Sulfate", 79.9568),
    ("Phosphate", 79.9663),
]


# ===========================================================================
# 1. Botanical Transformation Library Completeness Tests
# ===========================================================================

class TestBotanicalTransformationsLibrary:
    """Test completeness and precision of the botanical transformation matrix."""

    def test_all_16_transformations_present(self):
        for name, mass in VERIFIED_16_TRANSFORMATIONS:
            assert name in BOTANICAL_TRANSFORMATIONS, f"Missing transformation: {name}"
            assert BOTANICAL_TRANSFORMATIONS[name] == pytest.approx(mass, abs=1e-4)

    def test_delta_library_has_exact_keys(self):
        for name, mass in VERIFIED_16_TRANSFORMATIONS:
            assert mass in DELTA_LIBRARY, f"Mass {mass} for {name} missing from DELTA_LIBRARY"
            assert DELTA_LIBRARY[mass].lstrip("+-") == name

    def test_backward_compatibility_methyl_alias(self):
        """Verify legacy 14.0156 alias exists alongside 14.0157 for backward compatibility."""
        assert 14.0156 in DELTA_LIBRARY
        assert 14.0157 in DELTA_LIBRARY
        assert DELTA_LIBRARY[14.0156] == "+Methyl"
        assert DELTA_LIBRARY[14.0157] == "+Methyl"


# ===========================================================================
# 2. Bidirectional Delta Matching Tests
# ===========================================================================

class TestBidirectionalDeltaMatching:
    """Test positive (+Name) and negative (-Name) mass delta matching."""

    @pytest.mark.parametrize("name,mass", VERIFIED_16_TRANSFORMATIONS)
    def test_bidirectional_matching_all_16_transformations(self, name, mass):
        tmn = TransductiveMolecularNetwork(mass_tolerance_ppm=5.0)

        # Test positive delta (+mass)
        match_pos = tmn.match_delta(mass, tolerance_ppm=5.0)
        assert match_pos == f"+{name}", f"Expected +{name}, got {match_pos}"

        # Test negative delta (-mass)
        match_neg = tmn.match_delta(-mass, tolerance_ppm=5.0)
        assert match_neg == f"-{name}", f"Expected -{name}, got {match_neg}"

    def test_small_ppm_shifts_within_tolerance(self):
        tmn = TransductiveMolecularNetwork()
        hexose_mass = 162.0528
        # +3 ppm shift: 162.0528 * (1 + 3e-6)
        shifted_pos = hexose_mass * (1.0 + 3e-6)
        assert tmn.match_delta(shifted_pos, tolerance_ppm=5.0) == "+Hexose"

        # -3 ppm shift on negative delta
        shifted_neg = -hexose_mass * (1.0 - 3e-6)
        assert tmn.match_delta(shifted_neg, tolerance_ppm=5.0) == "-Hexose"


# ===========================================================================
# 3. Tolerance and Mathematical Orthogonality Tests
# ===========================================================================

class TestToleranceAndOrthogonality:
    """Test mathematical orthogonality: no cross-matching among closest pairs <= 5 ppm."""

    def test_glucuronide_vs_feruloyl_orthogonality(self):
        """Glucuronide (176.0321) vs Feruloyl (176.0473): delta 0.0152 Da (86.3 ppm separation)."""
        tmn = TransductiveMolecularNetwork()
        gluc_mass = 176.0321
        feru_mass = 176.0473

        # At 5.0 ppm, Glucuronide must never match Feruloyl
        assert tmn.match_delta(gluc_mass, tolerance_ppm=5.0) == "+Glucuronide"
        assert tmn.match_delta(feru_mass, tolerance_ppm=5.0) == "+Feruloyl"

        # Slight 3 ppm offset around Glucuronide
        assert tmn.match_delta(gluc_mass + 0.0003, tolerance_ppm=5.0) == "+Glucuronide"
        # Midpoint should match neither
        midpoint = (gluc_mass + feru_mass) / 2.0
        assert tmn.match_delta(midpoint, tolerance_ppm=5.0) is None

    def test_sulfate_vs_phosphate_orthogonality(self):
        """Sulfate (79.9568) vs Phosphate (79.9663): delta 0.0095 Da (118.8 ppm separation)."""
        tmn = TransductiveMolecularNetwork()
        sulf_mass = 79.9568
        phos_mass = 79.9663

        assert tmn.match_delta(sulf_mass, tolerance_ppm=5.0) == "+Sulfate"
        assert tmn.match_delta(phos_mass, tolerance_ppm=5.0) == "+Phosphate"
        assert tmn.match_delta(-sulf_mass, tolerance_ppm=5.0) == "-Sulfate"
        assert tmn.match_delta(-phos_mass, tolerance_ppm=5.0) == "-Phosphate"

    def test_caffeoyl_vs_hexose_orthogonality(self):
        """Caffeoyl (162.0317) vs Hexose (162.0528): delta 0.0211 Da (130.2 ppm separation)."""
        tmn = TransductiveMolecularNetwork()
        caff_mass = 162.0317
        hex_mass = 162.0528

        assert tmn.match_delta(caff_mass, tolerance_ppm=5.0) == "+Caffeoyl"
        assert tmn.match_delta(hex_mass, tolerance_ppm=5.0) == "+Hexose"

    def test_coumaroyl_vs_rhamnose_orthogonality(self):
        """Coumaroyl (146.0368) vs Rhamnose (146.0579): delta 0.0211 Da (144.5 ppm separation)."""
        tmn = TransductiveMolecularNetwork()
        coum_mass = 146.0368
        rham_mass = 146.0579

        assert tmn.match_delta(coum_mass, tolerance_ppm=5.0) == "+Coumaroyl"
        assert tmn.match_delta(rham_mass, tolerance_ppm=5.0) == "+Rhamnose"

    def test_delta_exceeding_tolerance_returns_none(self):
        tmn = TransductiveMolecularNetwork()
        # 10 ppm offset on Hexose (162.0528 * 10e-6 = 0.00162)
        far_delta = 162.0528 + 0.003
        assert tmn.match_delta(far_delta, tolerance_ppm=5.0) is None

        # Arbitrary non-matching mass
        assert tmn.match_delta(55.1234, tolerance_ppm=15.0) is None


# ===========================================================================
# 4. MS2 Fragment Peak Co-Validation Tests
# ===========================================================================

class TestFragmentPeakCoValidation:
    """Test MS2 fragment peak matching with mz_tolerance=0.02 Da."""

    def test_count_shared_peaks_1d(self):
        tmn = TransductiveMolecularNetwork()
        peaks1 = np.array([100.0, 150.0, 200.0, 250.0])
        # Peaks with 2 overlapping within 0.02 Da: 100.01 (diff 0.01), 200.015 (diff 0.015)
        # and 2 non-overlapping: 150.05 (diff 0.05), 300.0
        peaks2 = np.array([100.01, 150.05, 200.015, 300.0])

        shared = tmn.count_shared_peaks(peaks1, peaks2, mz_tolerance=0.02)
        assert shared == 2

    def test_count_shared_peaks_2d(self):
        tmn = TransductiveMolecularNetwork()
        # [mz, intensity] pairs
        peaks1 = np.array([[115.0, 100.0], [145.0, 80.0], [175.0, 50.0]])
        peaks2 = np.array([[115.01, 90.0], [145.005, 70.0], [175.018, 40.0]])

        shared = tmn.count_shared_peaks(peaks1, peaks2, mz_tolerance=0.02)
        assert shared == 3

    def test_count_shared_peaks_empty(self):
        tmn = TransductiveMolecularNetwork()
        empty = np.zeros((0, 2))
        peaks = np.array([[100.0, 10.0]])
        assert tmn.count_shared_peaks(empty, peaks) == 0
        assert tmn.count_shared_peaks(peaks, empty) == 0


# ===========================================================================
# 5. Scaffold Propagation Tests
# ===========================================================================

class TestScaffoldPropagation:
    """Test transductive scaffold propagation with directional and co-validation filters."""

    @pytest.fixture
    def setup_network(self) -> TransductiveMolecularNetwork:
        tmn = TransductiveMolecularNetwork(cosine_threshold=0.7, mass_tolerance_ppm=10.0)

        # Source spectrum: neutral mass 264.08, unit embedding
        rng = np.random.default_rng(42)
        vec_source = rng.standard_normal(128).astype(np.float32)
        vec_source = vec_source / np.linalg.norm(vec_source)
        peaks_source = np.array([100.0, 150.0, 200.0, 250.0])

        tmn.add_spectrum(
            spec_id="SOURCE_01",
            neutral_mass=264.08,
            embedding=vec_source,
            peaks=peaks_source,
        )

        # Target 1: mass gain +Hexose (264.08 + 162.0528 = 426.1328)
        # Vector highly similar to source (cos_sim ~ 0.99), 3 shared fragment peaks
        vec_t1 = vec_source + (rng.standard_normal(128).astype(np.float32) * 0.01)
        vec_t1 = vec_t1 / np.linalg.norm(vec_t1)
        peaks_t1 = np.array([100.005, 150.008, 200.01, 350.0])  # 3 shared peaks

        tmn.add_spectrum(
            spec_id="TARGET_GAIN",
            neutral_mass=426.1328,
            embedding=vec_t1,
            peaks=peaks_t1,
        )

        # Target 2: mass loss -Methyl (264.08 - 14.0157 = 250.0643)
        # Vector highly similar to source, 2 shared peaks
        vec_t2 = vec_source + (rng.standard_normal(128).astype(np.float32) * 0.01)
        vec_t2 = vec_t2 / np.linalg.norm(vec_t2)
        peaks_t2 = np.array([100.01, 200.015, 220.0, 230.0])  # 2 shared peaks

        tmn.add_spectrum(
            spec_id="TARGET_LOSS",
            neutral_mass=250.0643,
            embedding=vec_t2,
            peaks=peaks_t2,
        )

        # Target 3: mass match +Hexose, but only 1 shared peak (insufficient co-validation)
        tmn.add_spectrum(
            spec_id="TARGET_UNVALIDATED",
            neutral_mass=426.1328,
            embedding=vec_t1,
            peaks=np.array([100.005, 300.0, 400.0]),  # only 1 peak overlaps
        )

        # Target 4: low cosine similarity (< 0.7)
        vec_low = -vec_source
        tmn.add_spectrum(
            spec_id="TARGET_LOW_COS",
            neutral_mass=426.1328,
            embedding=vec_low,
            peaks=peaks_t1,
        )

        return tmn

    def test_propagate_scaffold_directional_true(self, setup_network: TransductiveMolecularNetwork):
        known = {"SOURCE_01": ("C15H12O5", "IK14SOURCE001")}

        # Test mass gain target
        cands_gain = setup_network.propagate_scaffold(
            target_id="TARGET_GAIN",
            known_scaffolds=known,
            min_shared_peaks=2,
            directional=True,
            tolerance_ppm=5.0,
        )
        assert len(cands_gain) == 1
        cand = cands_gain[0]
        assert cand.source_id == "SOURCE_01"
        assert cand.scaffold_smiles == "C15H12O5"
        assert cand.transformation == "+Hexose"
        assert cand.network_score > 0.7

        # Test mass loss target
        cands_loss = setup_network.propagate_scaffold(
            target_id="TARGET_LOSS",
            known_scaffolds=known,
            min_shared_peaks=2,
            directional=True,
            tolerance_ppm=5.0,
        )
        assert len(cands_loss) == 1
        assert cands_loss[0].transformation == "-Methyl"

    def test_propagate_scaffold_directional_false_legacy_default(
        self, setup_network: TransductiveMolecularNetwork
    ):
        """Verify directional=False preserves legacy unsigned delta matching (+Prefix)."""
        known = {"SOURCE_01": ("C15H12O5", "IK14SOURCE001")}

        cands_loss = setup_network.propagate_scaffold(
            target_id="TARGET_LOSS",
            known_scaffolds=known,
            min_shared_peaks=2,
            directional=False,
            tolerance_ppm=5.0,
        )
        assert len(cands_loss) == 1
        # With directional=False, abs(target - source) = 14.0157 -> "+Methyl"
        assert cands_loss[0].transformation == "+Methyl"

    def test_fragment_covalidation_filter(self, setup_network: TransductiveMolecularNetwork):
        known = {"SOURCE_01": ("C15H12O5", "IK14SOURCE001")}

        # TARGET_UNVALIDATED has only 1 shared fragment peak -> rejected when min_shared_peaks=2
        cands = setup_network.propagate_scaffold(
            target_id="TARGET_UNVALIDATED",
            known_scaffolds=known,
            min_shared_peaks=2,
        )
        assert len(cands) == 0

        # Accepted when min_shared_peaks=1
        cands_relaxed = setup_network.propagate_scaffold(
            target_id="TARGET_UNVALIDATED",
            known_scaffolds=known,
            min_shared_peaks=1,
        )
        assert len(cands_relaxed) == 1

    def test_cosine_threshold_filter(self, setup_network: TransductiveMolecularNetwork):
        known = {"SOURCE_01": ("C15H12O5", "IK14SOURCE001")}

        # TARGET_LOW_COS has negative cosine similarity -> rejected by cosine_threshold=0.7
        cands = setup_network.propagate_scaffold(
            target_id="TARGET_LOW_COS",
            known_scaffolds=known,
            min_shared_peaks=2,
        )
        assert len(cands) == 0

    def test_scaffold_propagation_edge_cases(self, setup_network: TransductiveMolecularNetwork):
        # Target not in network
        assert setup_network.propagate_scaffold("UNKNOWN_ID", {}) == []

        # Target == Source
        assert setup_network.propagate_scaffold("SOURCE_01", {"SOURCE_01": ("SMI", "IK")}) == []
