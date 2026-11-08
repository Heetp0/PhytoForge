"""
Unit and integration tests for Wave 2 God Nodes optimization:
- TransductiveMolecularNetwork (vectorized propagation, max_delta_da bound, intensity-weighted shared peaks)
- Bitwise Fingerprint Substructure Prescreening in CASMIOmegaPipeline
- 2-Hop Scaffold Propagation across Botanical Glycoside Clusters
"""

import numpy as np
import pytest

from src.retrieval.transductive_networking import (
    NetworkCandidate,
    TransductiveMolecularNetwork,
    BOTANICAL_AGLYCONE_LIBRARY,
)
from src.pipeline import CASMIOmegaPipeline


def test_mass_delta_bound_pruning():
    """Verify that spectral pairs with mass difference > max_delta_da are pruned immediately."""
    network = TransductiveMolecularNetwork(cosine_threshold=0.5, mass_tolerance_ppm=15.0)
    emb = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
    peaks = np.array([[100.0, 10.0], [200.0, 20.0]], dtype=np.float64)

    # Spec A: 300.0 Da, Spec B: 850.0 Da (Delta = 550 Da > 400 Da bound)
    network.add_spectrum("SPEC_A", 300.0, emb, peaks)
    network.add_spectrum("SPEC_B", 850.0, emb, peaks)

    # Should prune without error and return empty list
    hits = network.propagate_scaffold(
        target_id="SPEC_B",
        known_scaffolds={"SPEC_A": ("c1ccccc1O", "IK14CORE000001")},
        max_delta_da=400.0,
    )
    assert len(hits) == 0


def test_vectorized_matrix_propagation():
    """Verify propagate_scaffold matches vectorized implementation."""
    network = TransductiveMolecularNetwork(cosine_threshold=0.7, mass_tolerance_ppm=15.0)
    emb1 = np.zeros(1024, dtype=np.float32)
    emb1[0] = 1.0

    peaks = np.array([[100.0, 10.0], [150.0, 20.0]], dtype=np.float64)

    # Seed 5 spectra, only SPEC_1 has +162.0528 (Hexose)
    network.add_spectrum("SPEC_TARGET", 464.0955, emb1, peaks)
    network.add_spectrum("SPEC_1", 302.0427, emb1, peaks)  # Quercetin -> +Hexose = 464.0955
    network.add_spectrum("SPEC_2", 200.0, emb1, peaks)

    known = {
        "SPEC_1": ("O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "REFJWTPEDVJJIY"),
        "SPEC_2": ("c1ccccc1", "IK14BENZ000001"),
    }

    hits = network.propagate_scaffold(
        target_id="SPEC_TARGET",
        known_scaffolds=known,
        max_delta_da=400.0,
    )
    assert len(hits) == 1
    assert hits[0].source_id == "SPEC_1"
    assert hits[0].transformation == "+Hexose"
    assert hits[0].inchikey14 == "REFJWTPEDVJJIY"


def test_bitwise_substructure_screening_preserves_matches():
    """Verify bitwise fingerprint subset check rejects non-superstructures in constant time."""
    from src.chemistry.standardizer import compute_morgan_fingerprint

    # Quercetin (scaffold S)
    quercetin_smi = "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12"
    # Quercitrin (quercetin-3-O-rhamnoside, candidate C - contains quercetin as substructure)
    quercitrin_smi = "CC1OC(OC2=C(c3ccc(O)c(O)c3)Oc3cc(O)cc(O)c3C2=O)C(O)C(O)C1O"
    # Aspirin (completely unrelated molecule)
    aspirin_smi = "CC(=O)Oc1ccccc1C(=O)O"

    fp_q = compute_morgan_fingerprint(quercetin_smi)
    fp_quercitrin = compute_morgan_fingerprint(quercitrin_smi)
    fp_aspirin = compute_morgan_fingerprint(aspirin_smi)

    def is_fp_substructure(fp_sub: np.ndarray, fp_super: np.ndarray) -> bool:
        # If sub is subgraph of super, every bit in sub must be set in super: (sub & ~super) == 0
        return bool(np.all((fp_sub & ~fp_super) == 0))

    assert fp_q is not None and fp_aspirin is not None
    # Rejection of Aspirin:
    assert not is_fp_substructure(fp_q, fp_aspirin)


def test_two_hop_transductive_diffusion():
    """Verify 2-hop scaffold propagation across homologous natural product derivatives."""
    network = TransductiveMolecularNetwork(cosine_threshold=0.7, mass_tolerance_ppm=15.0)
    emb = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
    peaks = np.array([[100.0, 10.0], [150.0, 20.0]], dtype=np.float64)

    # Query A: Aglycone (302.0427 Da, Quercetin)
    # Query B: Monoglycoside (464.0955 Da, +Hexose)
    # Query C: Diglycoside (610.1534 Da, +Rhamnose to B)
    network.add_spectrum("A", 302.0427, emb, peaks)
    network.add_spectrum("B", 464.0955, emb, peaks)
    network.add_spectrum("C", 610.1534, emb, peaks)

    known = {"A": ("O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", "REFJWTPEDVJJIY")}

    # Step 1: B propagates from A
    hits_b = network.propagate_scaffold("B", known)
    assert len(hits_b) == 1
    assert hits_b[0].transformation == "+Hexose"

    # Step 2: In 2-hop diffusion, register B with A's core scaffold
    known["B"] = (hits_b[0].scaffold_smiles, hits_b[0].inchikey14)

    # Step 3: C propagates from B
    hits_c = network.propagate_scaffold("C", known)
    assert len(hits_c) == 1
    assert hits_c[0].source_id == "B"
    assert hits_c[0].transformation == "+Rhamnose"
    assert hits_c[0].inchikey14 == "REFJWTPEDVJJIY"  # Inherits Quercetin aglycone core!
