import numpy as np
import pytest
from src.retrieval.transductive_networking import (
    NetworkCandidate,
    TransductiveMolecularNetwork,
)


def test_pentose_and_hexose_delta_propagation():
    network = TransductiveMolecularNetwork(cosine_threshold=0.7)
    # Spec A has neutral mass 300.0
    # Spec B has neutral mass 432.0423 (+132.0423 pentose)
    # Spec C has neutral mass 462.0528 (+162.0528 hexose)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    shared_peaks = np.array([[100.0, 10.0], [150.0, 20.0]])
    network.add_spectrum(
        spec_id="S_A", neutral_mass=300.0, embedding=emb, peaks=shared_peaks
    )
    network.add_spectrum(
        spec_id="S_B", neutral_mass=432.0423, embedding=emb, peaks=shared_peaks
    )
    network.add_spectrum(
        spec_id="S_C", neutral_mass=462.0528, embedding=emb, peaks=shared_peaks
    )

    # Propagate solved scaffold from S_A to S_B (+Pentose)
    propagated_b = network.propagate_scaffold(
        target_id="S_B", known_scaffolds={"S_A": ("c1ccccc1O", "IK14CORE000001")}
    )
    assert len(propagated_b) == 1
    assert propagated_b[0].transformation == "+Pentose"
    assert propagated_b[0].source_id == "S_A"
    assert propagated_b[0].scaffold_smiles == "c1ccccc1O"
    assert propagated_b[0].inchikey14 == "IK14CORE000001"
    assert propagated_b[0].source_tier == "track_network"

    # Propagate solved scaffold from S_A to S_C (+Hexose)
    propagated_c = network.propagate_scaffold(
        target_id="S_C", known_scaffolds={"S_A": ("c1ccccc1O", "IK14CORE000001")}
    )
    assert len(propagated_c) == 1
    assert propagated_c[0].transformation == "+Hexose"
    assert propagated_c[0].source_id == "S_A"


def test_fragment_covalidation_rejects_insufficient_shared_peaks():
    network = TransductiveMolecularNetwork(cosine_threshold=0.7)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    # S_A and S_C have matching delta (+Glucuronide: 176.0321) but only 1 shared peak
    network.add_spectrum(
        spec_id="S_A",
        neutral_mass=300.0,
        embedding=emb,
        peaks=np.array([[100.0, 10.0], [150.0, 20.0]]),
    )
    network.add_spectrum(
        spec_id="S_C",
        neutral_mass=476.0321,
        embedding=emb,
        peaks=np.array([[100.0, 10.0], [220.0, 20.0]]),
    )
    propagated = network.propagate_scaffold(
        target_id="S_C", known_scaffolds={"S_A": ("c1ccccc1O", "IK14CORE000001")}
    )
    # Rejection because shared peaks == 1 (< 2 required)
    assert len(propagated) == 0


def test_all_seven_botanical_transformations():
    network = TransductiveMolecularNetwork(cosine_threshold=0.7, mass_tolerance_ppm=15.0)
    expected_transformations = {
        176.0321: "+Glucuronide",
        162.0528: "+Hexose",
        146.0579: "+Rhamnose",
        132.0423: "+Pentose",
        86.0004: "+Malonyl",
        42.0106: "+Acetyl",
        14.0156: "+Methyl",
    }
    for delta, name in expected_transformations.items():
        matched = network.match_delta(delta)
        assert matched == name, f"Delta {delta} should match {name}, got {matched}"


def test_low_cosine_rejection():
    network = TransductiveMolecularNetwork(cosine_threshold=0.7)
    emb_a = np.array([1.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0], dtype=np.float32)
    shared_peaks = np.array([[100.0, 10.0], [150.0, 20.0]])
    network.add_spectrum("S_A", 300.0, emb_a, shared_peaks)
    network.add_spectrum("S_B", 432.0423, emb_b, shared_peaks)

    propagated = network.propagate_scaffold(
        target_id="S_B", known_scaffolds={"S_A": ("c1ccccc1O", "IK14CORE000001")}
    )
    assert len(propagated) == 0


def test_target_not_found_and_empty_network_safety():
    network = TransductiveMolecularNetwork()
    assert network.propagate_scaffold("NONEXISTENT", {"S_A": ("c1ccccc1O", "IK14CORE000001")}) == []

    emb = np.array([1.0, 0.0], dtype=np.float32)
    peaks = np.array([[100.0, 10.0], [150.0, 20.0]])
    network.add_spectrum("S_A", 300.0, emb, peaks)
    # Target exists but known scaffold source does not exist
    assert network.propagate_scaffold("S_A", {"UNKNOWN_SRC": ("c1ccccc1O", "IK14CORE000001")}) == []
    # Self-propagation should be skipped
    assert network.propagate_scaffold("S_A", {"S_A": ("c1ccccc1O", "IK14CORE000001")}) == []


def test_ppm_tolerance_matching():
    network = TransductiveMolecularNetwork(mass_tolerance_ppm=15.0)
    ref_delta = 162.0528  # +Hexose

    # 10 ppm error should match
    within_delta = ref_delta * (1.0 + 10.0 / 1e6)
    assert network.match_delta(within_delta) == "+Hexose"

    # -10 ppm error should match
    within_delta_neg = ref_delta * (1.0 - 10.0 / 1e6)
    assert network.match_delta(within_delta_neg) == "+Hexose"

    # 25 ppm error should NOT match (exceeds 15.0 ppm limit)
    outside_delta = ref_delta * (1.0 + 25.0 / 1e6)
    assert network.match_delta(outside_delta) is None
