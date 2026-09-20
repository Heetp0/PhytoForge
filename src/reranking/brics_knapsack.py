"""BRICS Fragment Library and Substructure Decomposition for CASMI Track K.

This module decomposes chemical candidates into retrosynthetically tractable
fragments using RDKit's BRICS decomposition algorithm, strips attachment dummy
atoms ([*], [4*], etc.), sanitizes the resulting structures, and indexes them
by integer millimass units (round(neutral_loss_da * 10000)).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Attempt to load RDKit components with graceful fallback
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import BRICS, Descriptors

    # Silence verbose C++ logging during invalid SMILES parsing
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    Chem = None  # type: ignore
    RDLogger = None  # type: ignore
    BRICS = None  # type: ignore
    Descriptors = None  # type: ignore
    _RDKIT_AVAILABLE = False


def fragment_mass(smiles: str) -> float:
    """Calculate exact monoisotopic neutral mass of a fragment SMILES.

    Strips any RDKit BRICS dummy attachment atoms ([*], [4*], etc.) if present,
    sanitizes the molecule, and computes exact mass via Descriptors.ExactMolWt().
    Returns 0.0 on invalid SMILES, empty strings, unparseable structures, or
    if RDKit is unavailable.

    Args:
        smiles: Valid or dummy-annotated SMILES string.

    Returns:
        Exact monoisotopic neutral mass in Daltons as float, or 0.0 on failure.
    """
    if not _RDKIT_AVAILABLE or Chem is None or Descriptors is None:
        return 0.0

    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return 0.0

    s = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return 0.0

        # Check and remove dummy atoms (atomic number 0) in reverse index order
        rw = Chem.RWMol(mol)
        has_dummy = False
        for i in reversed(range(rw.GetNumAtoms())):
            atom = rw.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 0:
                rw.RemoveAtom(i)
                has_dummy = True

        clean_mol = rw.GetMol()
        if clean_mol.GetNumAtoms() == 0:
            return 0.0

        if has_dummy:
            Chem.SanitizeMol(clean_mol)

        mass = float(Descriptors.ExactMolWt(clean_mol))
        return mass if mass > 0.0 else 0.0
    except (ImportError, ModuleNotFoundError):
        return 0.0
    except Exception:
        return 0.0


def strip_dummy_atoms_and_sanitize(frag_smi: str) -> Optional[Tuple[str, float, int]]:
    """Clean a BRICS fragment SMILES by stripping dummy attachment atoms.

    Args:
        frag_smi: Fragment SMILES string produced by BRICSDecompose.

    Returns:
        Tuple of (clean_canonical_smiles, exact_mass, millimass_key), or None
        if the fragment cannot be cleaned, contains remaining dummy markers,
        or has non-positive mass.
    """
    if not _RDKIT_AVAILABLE or Chem is None or Descriptors is None:
        return None

    if not frag_smi or not isinstance(frag_smi, str) or not frag_smi.strip():
        return None

    try:
        frag_mol = Chem.MolFromSmiles(frag_smi.strip())
        if frag_mol is None:
            return None

        rw = Chem.RWMol(frag_mol)
        for i in reversed(range(rw.GetNumAtoms())):
            atom = rw.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 0:
                rw.RemoveAtom(i)

        clean_mol = rw.GetMol()
        if clean_mol.GetNumAtoms() == 0:
            return None

        Chem.SanitizeMol(clean_mol)
        clean_smi = Chem.MolToSmiles(clean_mol, canonical=True)

        # Output SMILES MUST contain NO dummy atom patterns ([*], [0*], [N*], etc.)
        if "*" in clean_smi or not clean_smi:
            return None

        # Verify clean_smi can be reparsed
        test_mol = Chem.MolFromSmiles(clean_smi)
        if test_mol is None:
            return None

        mass = float(Descriptors.ExactMolWt(clean_mol))
        if mass <= 0.0:
            return None

        millimass_key = int(round(mass * 10000))
        return clean_smi, mass, millimass_key
    except (ImportError, ModuleNotFoundError):
        raise
    except Exception:
        return None


def extract_fragments(
    smiles_list: List[str],
    min_mass: float = 0.0,
    max_mass: float = 2000.0,
) -> Dict[int, List[str]]:
    """Decompose molecules into clean BRICS fragments indexed by integer millimass.

    Uses RDKit BRICS decomposition guarded by try/except ImportError. Strips
    attachment dummy atoms ([*], [4*], etc.) and sanitizes all fragments.
    Degrades cleanly to an empty dict {} if RDKit is unavailable or raises
    ImportError.

    Args:
        smiles_list: List of candidate SMILES strings to decompose.
        min_mass: Optional minimum fragment mass cutoff in Daltons.
        max_mass: Optional maximum fragment mass cutoff in Daltons.

    Returns:
        Dict[int, List[str]] mapping integer millimass units (round(mass * 10000))
        to deduplicated lists of canonical sanitized SMILES strings.
    """
    if not _RDKIT_AVAILABLE or Chem is None or BRICS is None or Descriptors is None:
        return {}

    if not smiles_list:
        return {}

    library: Dict[int, List[str]] = {}
    seen_by_key: Dict[int, Set[str]] = {}

    for smi in smiles_list:
        if not smi or not isinstance(smi, str) or not smi.strip():
            continue

        s = smi.strip()
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue

            # Call BRICSDecompose guarded against ImportError
            raw_frags = BRICS.BRICSDecompose(mol, returnMols=False)
        except (ImportError, ModuleNotFoundError):
            return {}
        except Exception:
            continue

        for frag_smi in raw_frags:
            try:
                cleaned = strip_dummy_atoms_and_sanitize(frag_smi)
            except (ImportError, ModuleNotFoundError):
                return {}

            if cleaned is None:
                continue

            clean_smi, mass, millimass_key = cleaned
            if mass < min_mass or mass > max_mass:
                continue

            if millimass_key not in library:
                library[millimass_key] = []
                seen_by_key[millimass_key] = set()

            if clean_smi not in seen_by_key[millimass_key]:
                seen_by_key[millimass_key].add(clean_smi)
                library[millimass_key].append(clean_smi)

    return library


class _ExtractFragmentsDescriptor:
    """Descriptor enabling extract_fragments to be called on class or instance."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __get__(self, instance: Any, owner: Any) -> Callable:
        if instance is None:
            # Class-level call: BRICSFragmentLibrary.extract_fragments(smiles_list)
            def _class_wrapper(smiles_list: List[str], **kwargs: Any) -> Dict[int, List[str]]:
                return self.func(smiles_list, **kwargs)

            return _class_wrapper
        else:
            # Instance call: lib.extract_fragments(smiles_list)
            def _instance_wrapper(smiles_list: List[str], **kwargs: Any) -> Dict[int, List[str]]:
                if "min_mass" not in kwargs and hasattr(instance, "min_mass"):
                    kwargs["min_mass"] = instance.min_mass
                if "max_mass" not in kwargs and hasattr(instance, "max_mass"):
                    kwargs["max_mass"] = instance.max_mass
                result = self.func(smiles_list, **kwargs)
                if hasattr(instance, "_library"):
                    instance._library = result
                return result

            return _instance_wrapper


class BRICSFragmentLibrary:
    """Retrosynthetic BRICS fragment extraction and mass indexing library.

    Extracts fragments from molecule SMILES using RDKit BRICS decomposition,
    strips dummy attachment markers, and indexes resulting canonical structures
    by integer millimass units (round(neutral_mass * 10000)).
    """

    def __init__(
        self,
        min_mass: float = 0.0,
        max_mass: float = 2000.0,
    ) -> None:
        self.min_mass = float(min_mass)
        self.max_mass = float(max_mass)
        self._library: Dict[int, List[str]] = {}

    @property
    def library(self) -> Dict[int, List[str]]:
        """Access currently cached extracted library."""
        return self._library

    extract_fragments = _ExtractFragmentsDescriptor(extract_fragments)
    fragment_mass = staticmethod(fragment_mass)


__all__ = [
    "BRICSFragmentLibrary",
    "extract_fragments",
    "fragment_mass",
    "strip_dummy_atoms_and_sanitize",
]
