"""Bemis-Murcko scaffold extraction and zero-overlap cross-validation splitter."""

from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def get_scaffold(smiles: str, generic: bool = False) -> str:
    """
    Extract the Bemis-Murcko framework SMILES for a given molecule.

    Parameters
    ----------
    smiles : str
        Input SMILES representation of the molecule.
    generic : bool, default=False
        If True, converts all atoms to carbon and all bonds to single bonds
        using MurckoScaffold.MakeScaffoldGeneric (generic carbon framework).
        If False, preserves atom identities and aromaticity in the core ring system.

    Returns
    -------
    str
        Canonical SMILES of the Murcko scaffold. Returns empty string ("") for
        acyclic molecules or invalid empty inputs, and original stripped SMILES
        if RDKit is unavailable or parsing fails.
    """
    if smiles is None or not isinstance(smiles, (str, bytes)):
        return ""

    if isinstance(smiles, bytes):
        try:
            smiles = smiles.decode("utf-8")
        except Exception:
            return ""

    smiles_clean = str(smiles).strip()
    if not smiles_clean:
        return ""

    if not _RDKIT_AVAILABLE:
        return smiles_clean

    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return smiles_clean

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            return ""

        if generic:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)

        return Chem.MolToSmiles(scaffold_mol)
    except Exception:
        return smiles_clean


class BemisMurckoSplitter:
    """
    Bemis-Murcko Scaffold Cross-Validation Splitter.

    Partitions dataset rows into cross-validation folds grouped by molecular
    scaffold, mathematically guaranteeing strictly zero scaffold leakage between
    training and validation folds across all splits.

    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds. Must be >= 2.
    random_state : int, default=42
        Seed for the pseudo-random number generator used to break ties during
        greedy bin packing.
    generic : bool, default=False
        Whether to group molecules by generic carbon skeleton or exact Murcko scaffold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        generic: bool = False,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.random_state = random_state
        self.generic = generic

    def get_n_splits(
        self,
        df: Optional[pd.DataFrame] = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and validation sets.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing molecular records to split.
        smiles_col : str, default="smiles"
            Name of the column containing SMILES strings.

        Yields
        ------
        train_idx : np.ndarray
            The training set integer positional indices for that split.
        val_idx : np.ndarray
            The validation set integer positional indices for that split.
        """
        if smiles_col not in df.columns:
            raise KeyError(f"Column '{smiles_col}' not found in DataFrame.")

        n_samples = len(df)
        all_indices = np.arange(n_samples, dtype=int)

        if n_samples == 0:
            for _ in range(self.n_splits):
                yield np.empty(0, dtype=int), np.empty(0, dtype=int)
            return

        # Map each row's 0-based positional index to its scaffold
        smiles_list = df[smiles_col].tolist()
        scaffold_groups: Dict[str, List[int]] = {}
        for pos_idx, raw_smi in enumerate(smiles_list):
            smi_str = "" if pd.isna(raw_smi) else str(raw_smi)
            scaff = get_scaffold(smi_str, generic=self.generic)
            scaffold_groups.setdefault(scaff, []).append(pos_idx)

        # Order unique scaffolds: tie-break pseudo-randomly, then sort descending by group size
        rng = np.random.RandomState(self.random_state)
        unique_scaffolds = list(scaffold_groups.keys())
        rng.shuffle(unique_scaffolds)
        unique_scaffolds.sort(key=lambda s: len(scaffold_groups[s]), reverse=True)

        # Greedy bin packing into n_splits folds
        fold_indices: List[List[int]] = [[] for _ in range(self.n_splits)]
        fold_sizes = [0] * self.n_splits

        for scaff in unique_scaffolds:
            group = scaffold_groups[scaff]
            target_fold = int(np.argmin(fold_sizes))
            fold_indices[target_fold].extend(group)
            fold_sizes[target_fold] += len(group)

        # Yield train and validation index arrays
        for fold in range(self.n_splits):
            val_idx = np.array(sorted(fold_indices[fold]), dtype=int)
            train_idx = np.setdiff1d(all_indices, val_idx)
            yield train_idx, val_idx
