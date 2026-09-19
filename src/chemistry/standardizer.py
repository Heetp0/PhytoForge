"""
RDKit molecular standardization, salt stripping, charge neutralization,
tautomer canonicalization, and InChIKey14 extraction for Enveda CASMI 2026.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# Attempt to load RDKit components; allow graceful module load if RDKit is mid-install
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    # Silence verbose C++ logging from RDKit during parsing of malformed SMILES
    RDLogger.DisableLog("rdApp.*")

    # Module-level singletons for performance and thread safety
    _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
    _TAUTOMER_ENUMERATOR.SetMaxTautomers(100)  # Bound combinatorial explosion
    _UNCHARGER = rdMolStandardize.Uncharger()
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    _TAUTOMER_ENUMERATOR = None
    _UNCHARGER = None


def standardize_mol(smiles: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Standardizes a chemical structure from SMILES:
      1. Parses and sanitizes SMILES.
      2. Strips salts, solvents, and counterions to isolate the parent organic fragment.
      3. Neutralizes unquenched formal charges where chemically valid.
      4. Canonicalizes prototropic tautomers (e.g. 2-pyridone / 2-hydroxypyridine).
      5. Generates canonical SMILES and 14-character InChIKey14 (connectivity skeleton).

    Args:
        smiles: Input SMILES string.

    Returns:
        (canonical_smiles, inchikey14) tuple, or (None, None) if parsing/standardization fails.
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return None, None

    clean_smiles = smiles.strip()
    if len(clean_smiles) > 2000:
        return None, None

    if not _RDKIT_AVAILABLE:
        # Fallback hash if RDKit is not yet initialized
        import hashlib
        alphanumeric = "".join(c for c in clean_smiles if c.isalnum())
        ik14 = hashlib.sha256(alphanumeric.encode("utf-8")).hexdigest()[:14].upper()
        return clean_smiles, ik14

    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return None, None

        # Stage 1: Salt Stripping & Desolvation (retains largest organic parent)
        parent_mol = rdMolStandardize.FragmentParent(mol)
        if parent_mol is None:
            return None, None

        # Stage 2: Charge Neutralization
        neutral_mol = _UNCHARGER.uncharge(parent_mol)
        if neutral_mol is None:
            neutral_mol = parent_mol

        # Stage 3: Tautomer Canonicalization
        canonical_mol = _TAUTOMER_ENUMERATOR.Canonicalize(neutral_mol)
        if canonical_mol is None:
            canonical_mol = neutral_mol

        # Stage 4: Canonical SMILES and InChIKey14 Generation
        canonical_smiles = Chem.MolToSmiles(canonical_mol, canonical=True)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        if not inchikey or len(inchikey) < 14:
            return None, None

        inchikey14 = inchikey.split("-")[0]
        return canonical_smiles, inchikey14

    except Exception:
        # Fail safe on unanticipated C++ exceptions / invalid valence
        return None, None


def get_inchikey14(smiles_or_mol: Any) -> Optional[str]:
    """
    Directly extracts the canonical InChIKey14 from a SMILES string or RDKit Mol.
    Returns None on error.
    """
    if smiles_or_mol is None:
        return None

    if isinstance(smiles_or_mol, str):
        _, ik14 = standardize_mol(smiles_or_mol)
        return ik14

    if _RDKIT_AVAILABLE and isinstance(smiles_or_mol, Chem.Mol):
        try:
            parent = rdMolStandardize.FragmentParent(smiles_or_mol)
            neutral = _UNCHARGER.uncharge(parent) if parent else smiles_or_mol
            canon = _TAUTOMER_ENUMERATOR.Canonicalize(neutral) if neutral else smiles_or_mol
            inchikey = Chem.MolToInchiKey(canon)
            return inchikey.split("-")[0] if inchikey else None
        except Exception:
            return None

    return None


def deduplicate_candidates(candidates: List[Any]) -> List[Any]:
    """
    Deduplicates a ranked list of candidate objects by their `inchikey14` attribute.
    Retains the first occurrence (highest rank / score) and discards subsequent duplicates.
    
    Candidates with missing or empty `inchikey14` are omitted.

    Args:
        candidates: List of candidate objects having an `inchikey14` attribute.

    Returns:
        List of candidates with pairwise-distinct InChIKey14.
    """
    seen_inchikeys: Set[str] = set()
    deduped: List[Any] = []

    for cand in candidates:
        ik14 = getattr(cand, "inchikey14", None)
        if not ik14 or not isinstance(ik14, str):
            continue
        if ik14 not in seen_inchikeys:
            seen_inchikeys.add(ik14)
            deduped.append(cand)

    return deduped


def deduplicate_smiles_list(smiles_list: List[str]) -> List[Tuple[str, str]]:
    """
    Takes a raw ranked list of SMILES strings, standardizes each, and returns
    a deduplicated list of (canonical_smiles, inchikey14) preserving original ranking.
    """
    seen_inchikeys: Set[str] = set()
    unique_candidates: List[Tuple[str, str]] = []

    for smi in smiles_list:
        canon_smi, ik14 = standardize_mol(smi)
        if canon_smi and ik14 and ik14 not in seen_inchikeys:
            seen_inchikeys.add(ik14)
            unique_candidates.append((canon_smi, ik14))

    return unique_candidates
