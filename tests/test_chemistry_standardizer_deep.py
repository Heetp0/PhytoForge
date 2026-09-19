import pytest
from src.chemistry.standardizer import (
    standardize_mol,
    get_inchikey14,
    deduplicate_candidates,
    deduplicate_smiles_list,
    _RDKIT_AVAILABLE
)

def test_standardize_mol_none():
    assert standardize_mol(None) == (None, None)

def test_standardize_mol_empty():
    assert standardize_mol("") == (None, None)

def test_standardize_mol_whitespace():
    assert standardize_mol("   ") == (None, None)

def test_standardize_mol_non_string():
    assert standardize_mol(123) == (None, None)

def test_standardize_mol_invalid():
    assert standardize_mol("invalid_smiles_xyz") == (None, None)

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_standardize_mol_valid():
    smi, ik14 = standardize_mol("CC")
    assert smi == "CC"
    assert len(ik14) == 14

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_salt_stripping():
    smi, ik14 = standardize_mol("[Na+].CC(=O)[O-]")
    assert "Na" not in smi
    assert smi == "CC(=O)O"

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_charge_neutralization():
    smi, ik14 = standardize_mol("CC(=O)[O-]")
    assert smi == "CC(=O)O"

def test_get_inchikey14_none():
    assert get_inchikey14(None) is None

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_get_inchikey14_string():
    ik14 = get_inchikey14("CC")
    assert isinstance(ik14, str)
    assert len(ik14) == 14

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_get_inchikey14_rdkit_mol():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC")
    ik14 = get_inchikey14(mol)
    assert isinstance(ik14, str)
    assert len(ik14) == 14

def test_deduplicate_candidates_empty():
    assert deduplicate_candidates([]) == []

class DummyCandidate:
    def __init__(self, ik14):
        self.inchikey14 = ik14

def test_deduplicate_candidates_duplicates():
    c1 = DummyCandidate("ABCDEFGHIJKLMN")
    c2 = DummyCandidate("ABCDEFGHIJKLMN")
    c3 = DummyCandidate("ZYXWVUTSRQPONM")
    res = deduplicate_candidates([c1, c2, c3])
    assert len(res) == 2
    assert res[0] is c1
    assert res[1] is c3

def test_deduplicate_candidates_missing_attr():
    c1 = DummyCandidate("ABCDEFGHIJKLMN")
    class NoIk:
        pass
    c2 = NoIk()
    res = deduplicate_candidates([c1, c2])
    assert len(res) == 1
    assert res[0] is c1

def test_deduplicate_smiles_list_empty():
    assert deduplicate_smiles_list([]) == []

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_deduplicate_smiles_list_duplicates():
    smiles = ["CC", "CC", "CCO"]
    res = deduplicate_smiles_list(smiles)
    assert len(res) == 2
    smi_set = {x[0] for x in res}
    assert "CC" in smi_set
    assert "CCO" in smi_set

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_deduplicate_smiles_list_invalid():
    smiles = ["CC", "invalid_smi"]
    res = deduplicate_smiles_list(smiles)
    assert len(res) == 1
    assert res[0][0] == "CC"

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_inchikey14_format():
    smi, ik14 = standardize_mol("CCO")
    assert len(ik14) == 14
    assert ik14.isalnum()

@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
def test_tautomer_canonicalization():
    smi1, ik1_14 = standardize_mol("O=c1cccc[nH]1")
    smi2, ik2_14 = standardize_mol("Oc1ccccn1")
    assert ik1_14 == ik2_14

def test_no_rdkit_fallback(monkeypatch):
    import src.chemistry.standardizer as std
    monkeypatch.setattr(std, "_RDKIT_AVAILABLE", False)
    smi, ik14 = std.standardize_mol("CCO")
    assert smi == "CCO"
    assert len(ik14) == 14
    assert ik14.isupper()
    
    res = std.deduplicate_smiles_list(["CCO", "CCO"])
    assert len(res) == 1

def test_get_inchikey14_invalid_smiles():
    assert get_inchikey14("invalid_smiles") is None

def test_get_inchikey14_non_string():
    assert get_inchikey14(123) is None
    assert get_inchikey14(45.6) is None

def test_deduplicate_candidates_multiple_missing():
    class NoIk:
        pass
    res = deduplicate_candidates([NoIk(), NoIk()])
    assert len(res) == 0

def test_deduplicate_candidates_empty_string():
    c1 = DummyCandidate("")
    c2 = DummyCandidate(None)
    res = deduplicate_candidates([c1, c2])
    assert len(res) == 0

def test_deduplicate_smiles_list_all_invalid():
    assert deduplicate_smiles_list(["invalid1", "invalid2"]) == []

def test_fallback_hash_correctness(monkeypatch):
    import src.chemistry.standardizer as std
    monkeypatch.setattr(std, "_RDKIT_AVAILABLE", False)
    import hashlib
    smi = "CCO"
    expected_hash = hashlib.sha256(smi.encode("utf-8")).hexdigest()[:14].upper()
    _, ik14 = std.standardize_mol(smi)
    assert ik14 == expected_hash
