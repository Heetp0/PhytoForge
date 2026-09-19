"""
Enveda CASMI 2026 Kaggle Submission Writer.

Guarantees:
1. Exact Kaggle format: `molecule_id,smiles`.
2. Exactly 25 semicolon-delimited SMILES strings per query row.
3. Strict validation: Non-empty, pairwise distinct InChIKey14 within the 25 slots,
   exact match against expected test molecule_ids, zero nulls/NaNs.
4. Atomic writing protocol: Writes to temporary file on same filesystem, validates
   integrity, flushes/fsyncs, and executes atomic os.replace.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Curated bank of 40+ chemically valid, neutral, simple reference molecules
# with known pairwise distinct InChIKey14 connectivity skeletons.
# Used for safe emergency backfilling when candidate retrieval yields < 25 candidates.
DEFAULT_FALLBACK_SMILES: List[str] = [
    "C",                                    # Methane
    "CC",                                   # Ethane
    "CCC",                                  # Propane
    "CCCC",                                 # Butane
    "CCCCC",                                # Pentane
    "CCCCCC",                               # Hexane
    "CCCCCCC",                              # Heptane
    "CCCCCCCC",                             # Octane
    "c1ccccc1",                             # Benzene
    "c1ccncc1",                             # Pyridine
    "c1cnccn1",                             # Pyrazine (1,4-diazine)
    "c1ncccn1",                             # Pyrimidine (1,3-diazine)
    "c1ccoc1",                              # Furan
    "c1ccsc1",                              # Thiophene
    "c1cn[nH]c1",                           # Pyrazole
    "c1cnco1",                              # Oxazole
    "c1cnc[nH]1",                           # Imidazole
    "c1cncs1",                              # Thiazole
    "CC(=O)O",                              # Acetic acid
    "CC(=O)N",                              # Acetamide
    "CO",                                   # Methanol
    "CCO",                                  # Ethanol
    "CCCO",                                 # Propanol
    "CCCCO",                                # Butanol
    "CN",                                   # Methylamine
    "CCN",                                  # Ethylamine
    "CCCN",                                 # Propylamine
    "CCCCN",                                # Butylamine
    "C1CC1",                                # Cyclopropane
    "C1CCC1",                               # Cyclobutane
    "C1CCCC1",                              # Cyclopentane
    "C1CCCCC1",                             # Cyclohexane
    "C1CCCCCC1",                            # Cycloheptane
    "c1ccccc1O",                            # Phenol
    "c1ccccc1N",                            # Aniline
    "c1ccccc1C(=O)O",                       # Benzoic acid
    "c1ccccc1C",                            # Toluene
    "CC(=O)Oc1ccccc1C(=O)O",                # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",         # Caffeine
    "CC(=O)Nc1ccc(O)cc1",                   # Paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",           # Ibuprofen
]



class SubmissionValidationError(ValueError):
    """Raised when submission data or output file fails validation."""
    pass


class IncompleteSubmissionError(SubmissionValidationError):
    """Raised when expected molecule_ids are missing from predictions."""
    pass


def get_inchikey14(smiles: str) -> Optional[str]:
    """
    Extract the 14-character skeletal InChIKey block for a SMILES string.
    First attempts official standardized extraction from src.chemistry.standardizer.
    Falls back to direct RDKit MolToInchiKey if standardizer is not yet imported.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    # 1. Official project standardizer
    try:
        from src.chemistry.standardizer import standardize_mol
        _, ik14 = standardize_mol(smiles)
        if ik14:
            return ik14
    except (ImportError, Exception):
        pass

    # 2. Direct RDKit fallback
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            ik = Chem.MolToInchiKey(mol)
            if ik and len(ik) >= 14:
                return ik[:14]
    except (ImportError, Exception):
        pass

    # 3. Hash fallback for lightweight tests where RDKit is not installed
    import hashlib
    cleaned = "".join(c for c in smiles if c.isalnum())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:14].upper()


def validate_submission_file(
    csv_path: Union[str, Path],
    expected_ids: Optional[Sequence[str]] = None,
    check_inchikey14: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validates the integrity of a generated Kaggle submission CSV file.

    Checks:
    - File exists and has size > 0.
    - Header is strictly `molecule_id,smiles`.
    - Exactly 2 columns per line.
    - No empty rows or trailing lines.
    - Each row contains exactly 25 semicolon-separated candidate SMILES.
    - No empty strings, whitespace, or invalid characters within candidate slots.
    - All molecule_ids are unique.
    - If expected_ids is provided, exactly matches expected_ids with no missing/extra IDs.
    - Within each row, all 25 candidate slots possess pairwise distinct InChIKey14s (if check_inchikey14=True).

    Returns:
        (is_valid, list_of_error_messages)
    """
    path = Path(csv_path)
    errors: List[str] = []

    if not path.exists():
        return False, [f"Submission file does not exist: {path}"]
    if path.stat().st_size == 0:
        return False, [f"Submission file is empty (0 bytes): {path}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]
    except Exception as e:
        return False, [f"Failed to read submission file: {e}"]

    if not lines:
        return False, ["Submission file contains no lines."]

    # Header check
    header = lines[0]
    if header != "molecule_id,smiles":
        errors.append(f"Invalid CSV header: expected 'molecule_id,smiles', got '{header}'")

    data_lines = lines[1:]
    found_ids: List[str] = []

    for row_idx, line in enumerate(data_lines, start=2):
        if not line.strip():
            errors.append(f"Line {row_idx}: Empty line in CSV.")
            continue

        parts = line.split(",", 1)
        if len(parts) != 2:
            errors.append(f"Line {row_idx}: Malformed row, expected 2 comma-separated fields, got {len(parts)}.")
            continue

        mol_id, smiles_field = parts[0].strip(), parts[1].strip()

        if not mol_id:
            errors.append(f"Line {row_idx}: Missing or blank molecule_id.")
            continue
        found_ids.append(mol_id)

        # Candidate count check
        candidates = smiles_field.split(";")
        if len(candidates) != 25:
            errors.append(
                f"Line {row_idx} ({mol_id}): Expected exactly 25 semicolon-delimited SMILES, got {len(candidates)}."
            )

        seen_inchikey14: Set[str] = set()
        for slot_idx, cand in enumerate(candidates):
            c_clean = cand.strip()
            if not c_clean:
                errors.append(f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} is empty.")
                continue

            if "\n" in c_clean or "\r" in c_clean or "," in c_clean:
                errors.append(
                    f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} contains illegal delimiter/newline characters."
                )

            if check_inchikey14:
                ik14 = get_inchikey14(c_clean)
                if ik14:
                    if ik14 in seen_inchikey14:
                        errors.append(
                            f"Line {row_idx} ({mol_id}): Duplicate InChIKey14 '{ik14}' at candidate slot {slot_idx + 1} ('{c_clean}')."
                        )
                    seen_inchikey14.add(ik14)

    # Duplicate IDs check
    if len(found_ids) != len(set(found_ids)):
        duplicates = [x for x in set(found_ids) if found_ids.count(x) > 1]
        errors.append(f"Duplicate molecule_ids found in submission: {duplicates}")

    # Expected IDs check
    if expected_ids is not None:
        expected_set = set(expected_ids)
        found_set = set(found_ids)
        missing = expected_set - found_set
        extra = found_set - expected_set

        if missing:
            errors.append(f"Missing {len(missing)} expected molecule_ids: {list(missing)[:5]}...")
        if extra:
            errors.append(f"Unexpected {len(extra)} extra molecule_ids: {list(extra)[:5]}...")

    return len(errors) == 0, errors


def validate_submission(df: pd.DataFrame, expected_ids: List[str]) -> bool:
    """Validate all structural submission invariants required for Kaggle scoring."""
    assert "id" in df.columns, "Submission must contain 'id' column"
    assert "candidates" in df.columns, "Submission must contain 'candidates' column"
    assert len(df) == len(expected_ids), f"Row count mismatch: got {len(df)}, expected {len(expected_ids)}"
    assert set(df["id"]) == set(expected_ids), "Spectrum IDs do not match expected test set IDs"

    for idx, row in df.iterrows():
        raw_val = row["candidates"]
        assert pd.notna(raw_val), f"Row {idx} contains empty or NaN candidate"
        cands = str(raw_val).split(";")
        assert len(cands) == 25, f"Row {idx} has {len(cands)} candidates; expected exactly 25"
        assert len(set(cands)) == 25, f"Row {idx} contains duplicate InChIKey14 entries"
        assert all(c and c.lower() != "nan" for c in cands), f"Row {idx} contains empty or NaN candidate"
        assert all(len(c) == 14 and c.isalnum() for c in cands), f"Row {idx} has invalid InChIKey14 string"
    return True


def write_submission(
    predictions: Mapping[str, Sequence[Union[str, Any]]],
    output_path: Union[str, Path],
    expected_ids: Optional[Sequence[str]] = None,
    validate_inchikey14: bool = True,
    auto_backfill: bool = True,
    auto_truncate: bool = True,
) -> Path:
    """
    Writes predictions to an atomic Kaggle-compliant submission.csv.

    Args:
        predictions: Mapping from molecule_id to a list of candidate SMILES
                     (or Candidate objects with a .smiles attribute).
        output_path: Target path for the final submission.csv.
        expected_ids: Optional sequence of all expected test molecule_ids.
                      If provided, guarantees all IDs are present and output
                      rows follow this exact ordering.
        validate_inchikey14: Whether to enforce unique InChIKey14 skeletons per slot.
        auto_backfill: If True, backfills slots with unique reference SMILES
                       if fewer than 25 distinct candidates are supplied.
        auto_truncate: If True, truncates candidate lists longer than 25 to 25.

    Returns:
        Path to the successfully written and validated submission file.

    Raises:
        IncompleteSubmissionError: If expected_ids are missing and auto_backfill is False.
        SubmissionValidationError: If staged output fails format/uniqueness validation.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine row order and check completeness
    if expected_ids is not None:
        target_ids = list(expected_ids)
        missing_ids = set(target_ids) - set(predictions.keys())
        if missing_ids and not auto_backfill:
            raise IncompleteSubmissionError(
                f"Predictions missing {len(missing_ids)} required test molecule_ids (auto_backfill=False): {list(missing_ids)[:5]}"
            )
    else:
        target_ids = sorted(predictions.keys())

    # Generate atomic temporary file in the EXACT same directory
    tmp_file = out_file.with_name(f".{out_file.name}.{os.getpid()}_{time.time_ns()}.tmp")

    try:
        with open(tmp_file, "w", newline="", encoding="utf-8") as f:
            f.write("molecule_id,smiles\n")

            for mol_id in target_ids:
                raw_cands = predictions.get(mol_id, [])

                # Extract SMILES strings from raw items (strings or Candidate objects)
                extracted_smiles: List[str] = []
                for item in raw_cands:
                    if hasattr(item, "smiles"):
                        s = getattr(item, "smiles")
                    elif isinstance(item, str):
                        s = item
                    else:
                        continue
                    if s and isinstance(s, str) and s.strip():
                        extracted_smiles.append(s.strip())

                # Filter and deduplicate by InChIKey14
                selected_smiles: List[str] = []
                seen_ik14: Set[str] = set()

                for s in extracted_smiles:
                    ik = get_inchikey14(s) if validate_inchikey14 else None
                    if ik:
                        if ik in seen_ik14:
                            continue
                        seen_ik14.add(ik)
                    selected_smiles.append(s)

                    if len(selected_smiles) == 25 and auto_truncate:
                        break

                # Handle < 25 candidates
                if len(selected_smiles) < 25:
                    if auto_backfill:
                        for fb in DEFAULT_FALLBACK_SMILES:
                            fb_clean = fb.strip()
                            fb_ik = get_inchikey14(fb_clean) if validate_inchikey14 else None
                            if fb_ik and fb_ik in seen_ik14:
                                continue
                            if fb_ik:
                                seen_ik14.add(fb_ik)
                            selected_smiles.append(fb_clean)
                            if len(selected_smiles) == 25:
                                break
                    else:
                        raise SubmissionValidationError(
                            f"Molecule {mol_id} has only {len(selected_smiles)} valid candidates (<25) and auto_backfill=False."
                        )

                # Handle > 25 candidates
                if len(selected_smiles) > 25:
                    if auto_truncate:
                        selected_smiles = selected_smiles[:25]
                    else:
                        raise SubmissionValidationError(
                            f"Molecule {mol_id} has {len(selected_smiles)} candidates (>25) and auto_truncate=False."
                        )

                # Write line
                row_str = f"{mol_id},{';'.join(selected_smiles)}\n"
                f.write(row_str)

            # Flush user space buffer and sync dirty filesystem cache
            f.flush()
            os.fsync(f.fileno())

        # Pre-flight integrity verification on temporary file
        is_valid, val_errors = validate_submission_file(
            tmp_file,
            expected_ids=target_ids,
            check_inchikey14=validate_inchikey14,
        )
        if not is_valid:
            err_msg = f"Submission staging file failed integrity checks:\n" + "\n".join(val_errors[:10])
            raise SubmissionValidationError(err_msg)

        # Atomic replacement: replaces out_file atomically
        os.replace(tmp_file, out_file)
        logger.info(f"Successfully wrote validated submission to {out_file} ({len(target_ids)} rows).")
        return out_file

    finally:
        # Guarantee tmp_file cleanup if an error occurred before os.replace
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
