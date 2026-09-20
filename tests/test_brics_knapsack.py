"""Unit tests for BRICSFragmentLibrary and fragment mass calculation.

Verifies retrosynthetic fragment extraction, dummy atom ([*]) stripping,
integer millimass key encoding, multi-molecule deduplication, and graceful
fallback when RDKit is unavailable or raises ImportError.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rdkit import Chem

from src.reranking.brics_knapsack import (
    BRICSFragmentLibrary,
    extract_fragments,
    fragment_mass,
    strip_dummy_atoms_and_sanitize,
)


# Standard diverse natural product and drug molecules for testing
TEST_MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",  # Quercetin
    "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",  # Resveratrol
    "c1ccccc1C(=O)O",  # Benzoic acid
    "CCO",  # Ethanol
    "c1ccc2c(c1)ccc(=O)o2",  # Coumarin
    "CC(=O)c1ccc(O)cc1",  # 4-Hydroxyacetophenone
]


def test_dummy_atom_stripping_and_sanitization() -> None:
    """Verify dummy attachment atoms ([*], [0*], [4*], etc.) are completely stripped."""
    # Decompose Aspirin: produces [1*]C(C)=O, [16*]c1ccccc1[16*], [6*]C(=O)O, [3*]O[3*]
    aspirin_smiles = ["CC(=O)Oc1ccccc1C(=O)O"]
    library = extract_fragments(aspirin_smiles)

    assert len(library) > 0

    all_fragments = [frag for frag_list in library.values() for frag in frag_list]
    assert len(all_fragments) > 0

    for frag in all_fragments:
        # Must contain NO dummy atom patterns whatsoever
        assert "*" not in frag, f"Fragment {frag} contains '*' character"
        assert "[0*]" not in frag
        assert "[*]" not in frag
        assert "[1*]" not in frag
        assert "[16*]" not in frag

        # Must be valid, parseable canonical SMILES
        mol = Chem.MolFromSmiles(frag)
        assert mol is not None, f"Fragment {frag} could not be parsed by RDKit"
        assert mol.GetNumAtoms() > 0

        # Verify no atoms with atomic number 0 exist in the parsed molecule
        assert all(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms())

    # Verify expected clean fragments exist in the library
    assert "c1ccccc1" in all_fragments  # Benzene from [16*]c1ccccc1[16*]
    assert "CC=O" in all_fragments  # Acetaldehyde from [1*]C(C)=O
    assert "O=CO" in all_fragments  # Formic acid from [6*]C(=O)O
    assert "O" in all_fragments  # Water from [3*]O[3*]


def test_strip_dummy_atoms_and_sanitize_unit() -> None:
    """Directly test the strip_dummy_atoms_and_sanitize helper on synthetic fragments."""
    # Phenyl attachment
    res_phenyl = strip_dummy_atoms_and_sanitize("[4*]c1ccccc1")
    assert res_phenyl is not None
    smi, mass, key = res_phenyl
    assert smi == "c1ccccc1"
    assert pytest.approx(mass, abs=1e-4) == 78.04695
    assert key == 780470
    assert isinstance(key, int)

    # Carbonyl attachment
    res_acetyl = strip_dummy_atoms_and_sanitize("[1*]C(C)=O")
    assert res_acetyl is not None
    smi, mass, key = res_acetyl
    assert smi == "CC=O"
    assert pytest.approx(mass, abs=1e-4) == 44.02621
    assert key == 440262
    assert isinstance(key, int)

    # Carboxylic acid attachment
    res_acid = strip_dummy_atoms_and_sanitize("[6*]C(=O)O")
    assert res_acid is not None
    smi, mass, key = res_acid
    assert smi == "O=CO"
    assert pytest.approx(mass, abs=1e-4) == 46.00548
    assert key == 460055

    # Pure dummy atom -> should be None
    assert strip_dummy_atoms_and_sanitize("[*]") is None
    assert strip_dummy_atoms_and_sanitize("[1*][2*]") is None

    # Invalid SMILES -> should be None
    assert strip_dummy_atoms_and_sanitize("invalid_smiles_string") is None
    assert strip_dummy_atoms_and_sanitize("") is None
    assert strip_dummy_atoms_and_sanitize("   ") is None


def test_fragment_mass_known_molecules() -> None:
    """Verify fragment_mass accurately computes exact monoisotopic neutral mass."""
    # Benzene C6H6: 78.046950 Da
    assert pytest.approx(fragment_mass("c1ccccc1"), abs=1e-4) == 78.04695
    # Class-level and instance-level call
    assert pytest.approx(BRICSFragmentLibrary.fragment_mass("c1ccccc1"), abs=1e-4) == 78.04695
    lib = BRICSFragmentLibrary()
    assert pytest.approx(lib.fragment_mass("c1ccccc1"), abs=1e-4) == 78.04695

    # With dummy atom [4*]c1ccccc1: should strip dummy atom and calculate mass of benzene
    assert pytest.approx(fragment_mass("[4*]c1ccccc1"), abs=1e-4) == 78.04695

    # Acetic acid C2H4O2: 60.021129 Da
    assert pytest.approx(fragment_mass("CC(=O)O"), abs=1e-4) == 60.02113

    # Water H2O: 18.010565 Da
    assert pytest.approx(fragment_mass("O"), abs=1e-4) == 18.01056

    # Formic acid CH2O2: 46.005479 Da
    assert pytest.approx(fragment_mass("O=CO"), abs=1e-4) == 46.00548

    # Phenol C6H6O: 94.041865 Da
    assert pytest.approx(fragment_mass("Oc1ccccc1"), abs=1e-4) == 94.04186
    # Phenol with dummy atom [16*]c1ccc(O)cc1
    assert pytest.approx(fragment_mass("[16*]c1ccc(O)cc1"), abs=1e-4) == 94.04186


def test_fragment_mass_invalid_and_edge_inputs() -> None:
    """Verify fragment_mass safely returns 0.0 on malformed, empty, or unparseable inputs."""
    assert fragment_mass("invalid_smiles_string") == 0.0
    assert fragment_mass("C12345") == 0.0
    assert fragment_mass("") == 0.0
    assert fragment_mass("   ") == 0.0
    assert fragment_mass(None) == 0.0  # type: ignore[arg-type]
    assert fragment_mass(12345) == 0.0  # type: ignore[arg-type]
    assert fragment_mass("[*]") == 0.0
    assert fragment_mass("[1*][2*]") == 0.0


def test_extract_fragments_multi_molecule_corpus() -> None:
    """Test extract_fragments on multi-molecule corpus with integer millimass keys."""
    library = extract_fragments(TEST_MOLECULES)

    assert isinstance(library, dict)
    assert len(library) > 0

    for key, fragments in library.items():
        # Keys must strictly be integer millimass units
        assert isinstance(key, int), f"Key {key} is not an int (type: {type(key)})"
        assert key > 0, f"Key {key} must be strictly positive"

        # Values must be lists of strings
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        # Verify no duplicate SMILES in any key's fragment list
        assert len(fragments) == len(set(fragments)), f"Duplicates found in key {key}: {fragments}"

        for smi in fragments:
            assert isinstance(smi, str)
            assert "*" not in smi

            # Mass of fragment should match key within rounding
            mass = fragment_mass(smi)
            assert mass > 0.0
            expected_key = int(round(mass * 10000))
            assert expected_key == key, f"Expected key {expected_key} for {smi}, but got {key}"


def test_extract_fragments_deduplication() -> None:
    """Verify identical molecules and shared fragments are strictly deduplicated."""
    # Aspirin and Benzoic acid both yield phenyl ring (benzene c1ccccc1, key 780470)
    # Paracetamol and Aspirin both yield acetyl / acetaldehyde (CC=O, key 440262)
    corpus = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(=O)Oc1ccccc1C(=O)O",  # Duplicate Aspirin
        "c1ccccc1C(=O)O",  # Benzoic acid
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
    ]

    library = extract_fragments(corpus)

    # Benzene key 780470
    assert 780470 in library
    assert library[780470] == ["c1ccccc1"]  # Exactly one copy

    # Acetaldehyde key 440262
    assert 440262 in library
    assert library[440262] == ["CC=O"]  # Exactly one copy


def test_extract_fragments_empty_and_corrupted_inputs() -> None:
    """Verify extract_fragments gracefully handles empty and corrupt inputs."""
    assert extract_fragments([]) == {}
    assert extract_fragments(["", "invalid_smiles", "   "]) == {}
    assert extract_fragments([None, 12345]) == {}  # type: ignore[list-item]

    # Mixed valid and invalid inputs: invalid skipped, valid processed
    library = extract_fragments(["not_a_smiles", "c1ccccc1C(=O)O", ""])
    assert len(library) > 0
    assert 780470 in library
    assert "c1ccccc1" in library[780470]


def test_rdkit_mocked_import_error_graceful_degradation() -> None:
    """Verify clean degradation to empty dict {} when RDKit raises ImportError."""
    with patch("rdkit.Chem.BRICS.BRICSDecompose", side_effect=ImportError("Mocked missing RDKit")):
        result = extract_fragments(["CC(=O)Oc1ccccc1C(=O)O"])
        assert result == {}

    with patch("src.reranking.brics_knapsack._RDKIT_AVAILABLE", False):
        result = extract_fragments(["CC(=O)Oc1ccccc1C(=O)O"])
        assert result == {}

        mass = fragment_mass("c1ccccc1")
        assert mass == 0.0

    with patch("src.reranking.brics_knapsack.BRICS", None):
        result = extract_fragments(["CC(=O)Oc1ccccc1C(=O)O"])
        assert result == {}


def test_brics_fragment_library_class_and_instance_usage() -> None:
    """Verify BRICSFragmentLibrary works via both class-level and instance-level calls."""
    # Class call
    res_class = BRICSFragmentLibrary.extract_fragments(["c1ccccc1C(=O)O"])
    assert isinstance(res_class, dict)
    assert 780470 in res_class

    # Instance call
    lib = BRICSFragmentLibrary(min_mass=50.0, max_mass=500.0)
    res_inst = lib.extract_fragments(["CC(=O)Oc1ccccc1C(=O)O"])
    assert isinstance(res_inst, dict)
    assert lib.library == res_inst

    # With min_mass=50.0, water (MW 18.01) and acetaldehyde (MW 44.03) should be filtered out
    all_frags = [f for fl in res_inst.values() for f in fl]
    assert "O" not in all_frags
    assert "CC=O" not in all_frags
    assert "c1ccccc1" in all_frags


def test_complex_natural_product_chiral_decomposition() -> None:
    """Verify complex chiral natural products decompose cleanly without dummy remnants."""
    salicin = "OCC1OC(Oc2ccccc2CO)C(O)C(O)C1O"
    library = extract_fragments([salicin])

    assert len(library) > 0
    all_frags = [f for fl in library.values() for f in fl]

    # Every fragment must be free of dummy markers and sanitizable
    for f in all_frags:
        assert "*" not in f
        mol = Chem.MolFromSmiles(f)
        assert mol is not None
        assert mol.GetNumAtoms() > 0


def test_fragment_mass_multi_dummy_indices() -> None:
    """Verify fragment_mass correctly strips fragments with multiple dummy attachment atoms."""
    # Resorcinol with 2 attachments: [16*]c1cc(O)cc(O)c1
    res = fragment_mass("[16*]c1cc(O)cc(O)c1")
    assert pytest.approx(res, abs=1e-4) == 110.0368

    # Acetaldehyde with attachment: [1*]C(C)=O
    res_acetyl = fragment_mass("[1*]C(C)=O")
    assert pytest.approx(res_acetyl, abs=1e-4) == 44.02621

    # Disubstituted benzene: [16*]c1ccc([16*])cc1
    res_benz = fragment_mass("[16*]c1ccc([16*])cc1")
    assert pytest.approx(res_benz, abs=1e-4) == 78.04695

