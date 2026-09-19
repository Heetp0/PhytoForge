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
import numpy as np

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
    output_format: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates the integrity of a generated Kaggle submission CSV file.

    Supports both formats:
    - 'inchikey14': header 'id,candidates', 25 semicolon-delimited 14-char alphanumeric InChIKeys.
    - 'smiles': header 'molecule_id,smiles', 25 semicolon-delimited valid SMILES strings.
    If output_format is None, the format is auto-detected from the CSV header.

    Returns:
        (is_valid, list_of_error_messages)
    """
    path = Path(csv_path)
    errors: List[str] = []

    if not path.exists():
        return False, [f"Submission file does not exist: {path}"]
    if path.is_dir():
        return False, [f"Submission path is a directory, not a file: {path}"]
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
    header = lines[0].strip()
    if output_format is not None:
        fmt = str(output_format).strip().lower()
        if fmt == "inchikey14":
            if header != "id,candidates":
                errors.append(f"Invalid CSV header: expected 'id,candidates', got '{header}'")
        elif fmt == "smiles":
            if header != "molecule_id,smiles":
                errors.append(f"Invalid CSV header: expected 'molecule_id,smiles', got '{header}'")
        else:
            errors.append(f"Unknown format '{output_format}'. Must be 'inchikey14' or 'smiles'")
            return False, errors
    else:
        if header == "id,candidates":
            fmt = "inchikey14"
        elif header == "molecule_id,smiles":
            fmt = "smiles"
        else:
            errors.append(f"Invalid CSV header: expected 'molecule_id,smiles' or 'id,candidates', got '{header}'")
            return False, errors

    data_lines = lines[1:]
    found_ids: List[str] = []

    for row_idx, line in enumerate(data_lines, start=2):
        if not line.strip():
            errors.append(f"Line {row_idx}: Empty line in CSV.")
            continue

        try:
            parsed_row = next(csv.reader([line]))
        except Exception:
            parsed_row = None

        if parsed_row and len(parsed_row) == 2:
            mol_id, cands_field = parsed_row[0].strip(), parsed_row[1].strip()
        else:
            parts = line.split(",", 1)
            if len(parts) != 2:
                errors.append(f"Line {row_idx}: Malformed row, expected 2 comma-separated fields, got {len(parts)}.")
                continue
            mol_id, cands_field = parts[0].strip(), parts[1].strip()
            if cands_field.startswith('"') and cands_field.endswith('"'):
                cands_field = cands_field[1:-1].strip()

        if not mol_id:
            errors.append(f"Line {row_idx}: Missing or blank molecule_id/id.")
            continue
        found_ids.append(mol_id)

        # Candidate count check
        candidates = cands_field.split(";")
        if len(candidates) != 25:
            errors.append(
                f"Line {row_idx} ({mol_id}): Expected exactly 25 semicolon-delimited candidates, got {len(candidates)}."
            )

        if fmt == "inchikey14":
            seen_ik14: Set[str] = set()
            for slot_idx, cand in enumerate(candidates):
                if not cand or not cand.strip():
                    errors.append(f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} is empty.")
                    continue
                if cand != cand.strip():
                    errors.append(
                        f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} ('{cand}') has leading/trailing whitespace."
                    )
                c_clean = cand.strip()
                if len(c_clean) != 14 or not c_clean.isalnum():
                    errors.append(
                        f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} ('{c_clean}') is not a valid 14-char alphanumeric InChIKey14."
                    )
                if c_clean in seen_ik14:
                    errors.append(
                        f"Line {row_idx} ({mol_id}): Duplicate InChIKey14 '{c_clean}' at candidate slot {slot_idx + 1}."
                    )
                seen_ik14.add(c_clean)
        else:  # smiles
            seen_smiles: Set[str] = set()
            seen_inchikey14: Set[str] = set()
            for slot_idx, cand in enumerate(candidates):
                if not cand or not cand.strip():
                    errors.append(f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} is empty.")
                    continue
                if cand != cand.strip():
                    errors.append(
                        f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} ('{cand}') has leading/trailing whitespace."
                    )
                c_clean = cand.strip()

                if any(ch in c_clean for ch in (" ", "\t", "\n", "\r", ",")):
                    errors.append(
                        f"Line {row_idx} ({mol_id}): Candidate slot {slot_idx + 1} contains illegal delimiter/whitespace characters."
                    )

                if c_clean in seen_smiles:
                    errors.append(
                        f"Line {row_idx} ({mol_id}): Duplicate SMILES '{c_clean}' at candidate slot {slot_idx + 1}."
                    )
                seen_smiles.add(c_clean)

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
        expected_str_list = [str(x).strip() for x in expected_ids]
        expected_set = set(expected_str_list)
        found_set = set(found_ids)
        missing = expected_set - found_set
        extra = found_set - expected_set

        if len(expected_str_list) != len(expected_set):
            dup_expected = [x for x in expected_set if expected_str_list.count(x) > 1]
            errors.append(f"Duplicate expected IDs provided: {dup_expected}")

        if len(found_ids) != len(expected_str_list):
            errors.append(f"Row count mismatch: found {len(found_ids)} rows, expected {len(expected_str_list)} rows.")

        if missing:
            errors.append(f"Missing {len(missing)} expected molecule_ids: {list(missing)[:5]}...")
        if extra:
            errors.append(f"Unexpected {len(extra)} extra molecule_ids: {list(extra)[:5]}...")

    return len(errors) == 0, errors


def validate_submission(
    df: pd.DataFrame,
    expected_ids: Optional[Sequence[str]] = None,
    output_format: Optional[str] = None,
) -> bool:
    """Validate all structural submission invariants required for Kaggle scoring.
    
    Supports both:
    - 'inchikey14': columns 'id' and 'candidates' (25 semicolon-delimited 14-char alphanumeric InChIKeys)
    - 'smiles': columns 'molecule_id' and 'smiles' (25 semicolon-delimited valid SMILES)
    If output_format is None, auto-detects from DataFrame columns.
    """
    if output_format is not None:
        fmt = str(output_format).strip().lower()
        if fmt not in ("inchikey14", "smiles"):
            raise ValueError(f"Invalid output_format '{output_format}'. Must be 'inchikey14' or 'smiles'.")
    else:
        if "id" in df.columns and "candidates" in df.columns:
            fmt = "inchikey14"
        elif "molecule_id" in df.columns and "smiles" in df.columns:
            fmt = "smiles"
        elif "id" in df.columns:
            fmt = "inchikey14"
        elif "molecule_id" in df.columns:
            fmt = "smiles"
        elif "candidates" in df.columns:
            fmt = "inchikey14"
        elif "smiles" in df.columns:
            fmt = "smiles"
        else:
            fmt = "inchikey14"

    if fmt == "inchikey14":
        assert "id" in df.columns, "Submission must contain 'id' column"
        assert "candidates" in df.columns, "Submission must contain 'candidates' column"
        assert list(df.columns) == ["id", "candidates"], f"Submission columns must be strictly ['id', 'candidates'], got {list(df.columns)}"
        id_col = "id"
        cand_col = "candidates"
    else:  # smiles
        assert "molecule_id" in df.columns, "Submission must contain 'molecule_id' column"
        assert "smiles" in df.columns, "Submission must contain 'smiles' column"
        assert list(df.columns) == ["molecule_id", "smiles"], f"Submission columns must be strictly ['molecule_id', 'smiles'], got {list(df.columns)}"
        id_col = "molecule_id"
        cand_col = "smiles"

    assert df[id_col].notna().all(), f"Submission '{id_col}' column contains NaN/null values"
    found_ids = [str(x).strip() for x in df[id_col]]
    assert all(x and x.lower() != "nan" for x in found_ids), f"Submission '{id_col}' column contains blank or NaN IDs"
    assert len(found_ids) == len(set(found_ids)), f"Duplicate {id_col} entries found in submission: {[x for x in set(found_ids) if found_ids.count(x) > 1]}"

    if expected_ids is not None:
        expected_str_ids = [str(x).strip() for x in expected_ids]
        assert len(expected_str_ids) == len(set(expected_str_ids)), f"Duplicate expected IDs provided: {[x for x in set(expected_str_ids) if expected_str_ids.count(x) > 1]}"
        assert len(df) == len(expected_str_ids), f"Row count mismatch: got {len(df)}, expected {len(expected_str_ids)}"
        assert set(found_ids) == set(expected_str_ids), "Spectrum IDs do not match expected test set IDs"

    for idx, row in df.iterrows():
        raw_val = row[cand_col]
        if isinstance(raw_val, (list, tuple, np.ndarray, pd.Series)):
            cands = [str(x) for x in raw_val]
        else:
            assert not (pd.isna(raw_val) if np.isscalar(raw_val) else False), f"Row {idx} contains empty or NaN candidate"
            cands = str(raw_val).split(";")
        assert len(cands) == 25, f"Row {idx} has {len(cands)} candidates; expected exactly 25"
        assert all(c and c.strip() and c.lower() != "nan" for c in cands), f"Row {idx} contains empty or NaN candidate"
        assert all(c == c.strip() for c in cands), f"Row {idx} contains candidate with leading/trailing whitespace"
        if fmt == "inchikey14":
            assert len(set(cands)) == 25, f"Row {idx} contains duplicate InChIKey14 entries"
            assert all(len(c) == 14 and c.isalnum() for c in cands), f"Row {idx} has invalid InChIKey14 string"
        else:
            assert len(set(cands)) == 25, f"Row {idx} contains duplicate SMILES entries"
            assert not any(any(ch in c for ch in (" ", "\t", "\n", "\r", ",")) for c in cands), f"Row {idx} contains illegal delimiter/whitespace in SMILES"

    return True


def write_submission(
    predictions: Union[Mapping[str, Sequence[Union[str, Any]]], pd.DataFrame],
    output_path: Union[str, Path],
    expected_ids: Optional[Sequence[str]] = None,
    validate_inchikey14: bool = True,
    auto_backfill: bool = True,
    auto_truncate: bool = True,
    output_format: Optional[str] = None,
) -> Path:
    """
    Writes predictions to an atomic Kaggle-compliant submission.csv.

    Args:
        predictions: Mapping from spectrum/molecule ID to candidate items OR a pre-formed pd.DataFrame.
        output_path: Target path for the final submission.csv.
        expected_ids: Optional sequence of all expected test molecule/spectrum IDs.
                      If provided, guarantees all IDs are present and output
                      rows follow this exact ordering.
        validate_inchikey14: Whether to enforce unique InChIKey14 skeletons per slot.
        auto_backfill: If True, backfills slots with unique reference candidates
                       if fewer than 25 distinct candidates are supplied.
        auto_truncate: If True, truncates candidate lists longer than 25 to 25.
        output_format: Target format ('inchikey14' or 'smiles'). If None, inferred
                       from DataFrame columns or defaults to 'smiles' for mapping inputs.

    Returns:
        Path to the successfully written and validated submission file.

    Raises:
        IncompleteSubmissionError: If expected_ids are missing and auto_backfill is False.
        SubmissionValidationError: If staged output fails format/uniqueness validation.
    """
    out_file = Path(output_path).resolve()
    if out_file.exists() and out_file.is_dir():
        raise SubmissionValidationError(f"Target output path '{out_file}' is an existing directory, not a file.")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if output_format is not None:
        fmt = str(output_format).strip().lower()
        if fmt not in ("inchikey14", "smiles"):
            raise ValueError(f"Invalid output_format '{output_format}'. Must be 'inchikey14' or 'smiles'.")
    elif isinstance(predictions, pd.DataFrame):
        if "candidates" in predictions.columns:
            fmt = "inchikey14"
        elif "smiles" in predictions.columns:
            fmt = "smiles"
        elif "id" in predictions.columns and "molecule_id" not in predictions.columns:
            fmt = "inchikey14"
        elif "molecule_id" in predictions.columns:
            fmt = "smiles"
        else:
            fmt = "inchikey14"
    else:
        fmt = "smiles"

    # Normalize predictions into an id -> candidate items dictionary
    if isinstance(predictions, pd.DataFrame):
        if predictions.empty and len(predictions.columns) == 0:
            if expected_ids is not None and auto_backfill:
                preds_dict = {str(k).strip(): [] for k in expected_ids}
                target_ids_from_df = [str(k).strip() for k in expected_ids]
            else:
                preds_dict = {}
                target_ids_from_df = []
        else:
            has_id = "id" in predictions.columns or "molecule_id" in predictions.columns
            has_cand = "candidates" in predictions.columns or "smiles" in predictions.columns
            if len(predictions.columns) < 2 and not (has_id and has_cand):
                raise SubmissionValidationError(
                    f"DataFrame predictions must have at least 2 columns (ID and candidates), got {len(predictions.columns)} column(s): {list(predictions.columns)}"
                )

            if fmt == "inchikey14":
                id_col = "id" if "id" in predictions.columns else ("molecule_id" if "molecule_id" in predictions.columns else predictions.columns[0])
                cands_col = "candidates" if "candidates" in predictions.columns else ("smiles" if "smiles" in predictions.columns else predictions.columns[1])
            else:
                id_col = "molecule_id" if "molecule_id" in predictions.columns else ("id" if "id" in predictions.columns else predictions.columns[0])
                cands_col = "smiles" if "smiles" in predictions.columns else ("candidates" if "candidates" in predictions.columns else predictions.columns[1])

            preds_dict: Dict[str, List[Any]] = {}
            target_ids_from_df: List[str] = []
            for _, row in predictions.iterrows():
                raw_id = row[id_col]
                if pd.isna(raw_id) or not str(raw_id).strip() or str(raw_id).strip().lower() == "nan":
                    raise SubmissionValidationError(f"Row {len(target_ids_from_df)} has invalid or NaN ID in column '{id_col}'.")
                m_id = str(raw_id).strip()
                target_ids_from_df.append(m_id)
                val = row[cands_col]
                if isinstance(val, (list, tuple, np.ndarray, pd.Series, Sequence)) and not isinstance(val, (str, bytes)):
                    preds_dict[m_id] = list(val)
                elif isinstance(val, str):
                    preds_dict[m_id] = [x.strip() for x in val.split(";") if x.strip()]
                elif val is None or (np.isscalar(val) and pd.isna(val)):
                    preds_dict[m_id] = []
                else:
                    preds_dict[m_id] = [val]
    else:
        preds_dict = {
            str(k).strip(): list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series, Sequence)) and not isinstance(v, (str, bytes)) else [v]
            for k, v in predictions.items()
        }
        if any(not k or k.lower() == "nan" for k in preds_dict.keys()):
            raise SubmissionValidationError("Predictions mapping contains blank or NaN molecule/spectrum IDs.")
        target_ids_from_df = []

    if expected_ids is not None:
        target_ids = [str(x).strip() for x in expected_ids]
        missing_ids = set(target_ids) - set(preds_dict.keys())
        if missing_ids and not auto_backfill:
            raise IncompleteSubmissionError(
                f"Predictions missing {len(missing_ids)} required test molecule_ids (auto_backfill=False): {list(missing_ids)[:5]}"
            )
    else:
        if isinstance(predictions, pd.DataFrame):
            target_ids = target_ids_from_df
        else:
            target_ids = sorted(preds_dict.keys())

    if len(target_ids) != len(set(target_ids)):
        duplicates = [x for x in set(target_ids) if target_ids.count(x) > 1]
        raise SubmissionValidationError(f"Duplicate molecule_ids found in submission: {duplicates}")

    # Generate atomic temporary file in the EXACT same directory
    tmp_file = out_file.with_name(f".{out_file.name}.{os.getpid()}_{time.time_ns()}.tmp")

    try:
        with open(tmp_file, "w", newline="", encoding="utf-8") as f:
            if fmt == "inchikey14":
                f.write("id,candidates\n")
                fallback_pool = [f"IK14FALL{i:06d}" for i in range(50)]

                for mol_id in target_ids:
                    raw_cands = preds_dict.get(mol_id, [])
                    extracted_iks: List[str] = []
                    for item in raw_cands:
                        s = None
                        if hasattr(item, "inchikey14") and getattr(item, "inchikey14"):
                            s = getattr(item, "inchikey14")
                        elif isinstance(item, dict) and item.get("inchikey14"):
                            s = item["inchikey14"]
                        elif isinstance(item, str):
                            s = item
                        elif hasattr(item, "smiles") and getattr(item, "smiles"):
                            s = get_inchikey14(getattr(item, "smiles"))
                        elif isinstance(item, dict) and item.get("smiles"):
                            s = get_inchikey14(item["smiles"])

                        if s and isinstance(s, str) and s.strip():
                            s_clean = s.strip()
                            if len(s_clean) == 14 and s_clean.isalnum():
                                ik = s_clean
                            else:
                                ik = get_inchikey14(s_clean)
                            if ik:
                                extracted_iks.append(ik)

                    selected_iks: List[str] = []
                    seen_ik14: Set[str] = set()

                    for ik in extracted_iks:
                        if ik in seen_ik14:
                            continue
                        seen_ik14.add(ik)
                        selected_iks.append(ik)
                        if len(selected_iks) == 25 and auto_truncate:
                            break

                    if len(selected_iks) < 25:
                        if auto_backfill:
                            for fb in fallback_pool:
                                if fb not in seen_ik14:
                                    seen_ik14.add(fb)
                                    selected_iks.append(fb)
                                    if len(selected_iks) == 25:
                                        break
                            pad_idx = 0
                            while len(selected_iks) < 25:
                                dummy = f"PAD{pad_idx:011d}"
                                if dummy not in seen_ik14:
                                    seen_ik14.add(dummy)
                                    selected_iks.append(dummy)
                                pad_idx += 1
                        else:
                            raise SubmissionValidationError(
                                f"Molecule {mol_id} has only {len(selected_iks)} valid candidates (<25) and auto_backfill=False."
                            )

                    if len(selected_iks) > 25:
                        if auto_truncate:
                            selected_iks = selected_iks[:25]
                        else:
                            raise SubmissionValidationError(
                                f"Molecule {mol_id} has {len(selected_iks)} candidates (>25) and auto_truncate=False."
                            )

                    row_str = f"{mol_id},{';'.join(selected_iks)}\n"
                    f.write(row_str)

            else:  # smiles
                f.write("molecule_id,smiles\n")

                for mol_id in target_ids:
                    raw_cands = preds_dict.get(mol_id, [])
                    extracted_smiles: List[str] = []
                    for item in raw_cands:
                        s = None
                        if hasattr(item, "smiles") and getattr(item, "smiles"):
                            s = getattr(item, "smiles")
                        elif hasattr(item, "scaffold_smiles") and getattr(item, "scaffold_smiles"):
                            s = getattr(item, "scaffold_smiles")
                        elif isinstance(item, dict) and item.get("smiles"):
                            s = item["smiles"]
                        elif isinstance(item, dict) and item.get("scaffold_smiles"):
                            s = item["scaffold_smiles"]
                        elif isinstance(item, str):
                            s = item

                        if s and isinstance(s, str) and s.strip():
                            s_clean = s.strip()
                            if any(ch in s_clean for ch in (" ", "\t", "\n", "\r", ",")):
                                continue
                            extracted_smiles.append(s_clean)

                    selected_smiles: List[str] = []
                    seen_smiles: Set[str] = set()
                    seen_ik14: Set[str] = set()

                    for s in extracted_smiles:
                        if s in seen_smiles:
                            continue
                        ik = get_inchikey14(s) if validate_inchikey14 else None
                        if ik:
                            if ik in seen_ik14:
                                continue
                            seen_ik14.add(ik)
                        seen_smiles.add(s)
                        selected_smiles.append(s)

                        if len(selected_smiles) == 25 and auto_truncate:
                            break

                    if len(selected_smiles) < 25:
                        if auto_backfill:
                            for fb in DEFAULT_FALLBACK_SMILES:
                                fb_clean = fb.strip()
                                if fb_clean in seen_smiles:
                                    continue
                                fb_ik = get_inchikey14(fb_clean) if validate_inchikey14 else None
                                if fb_ik and fb_ik in seen_ik14:
                                    continue
                                if fb_ik:
                                    seen_ik14.add(fb_ik)
                                seen_smiles.add(fb_clean)
                                selected_smiles.append(fb_clean)
                                if len(selected_smiles) == 25:
                                    break

                            alkane_k = 1
                            while len(selected_smiles) < 25:
                                cand_smi = "C" * alkane_k
                                if cand_smi not in seen_smiles:
                                    cand_ik = get_inchikey14(cand_smi) if validate_inchikey14 else None
                                    if not cand_ik or cand_ik not in seen_ik14:
                                        seen_smiles.add(cand_smi)
                                        if cand_ik:
                                            seen_ik14.add(cand_ik)
                                        selected_smiles.append(cand_smi)
                                alkane_k += 1
                        else:
                            raise SubmissionValidationError(
                                f"Molecule {mol_id} has only {len(selected_smiles)} valid candidates (<25) and auto_backfill=False."
                            )

                    if len(selected_smiles) > 25:
                        if auto_truncate:
                            selected_smiles = selected_smiles[:25]
                        else:
                            raise SubmissionValidationError(
                                f"Molecule {mol_id} has {len(selected_smiles)} candidates (>25) and auto_truncate=False."
                            )

                    row_str = f"{mol_id},{';'.join(selected_smiles)}\n"
                    f.write(row_str)

            f.flush()
            os.fsync(f.fileno())

        # Pre-flight integrity verification on temporary file
        is_valid, val_errors = validate_submission_file(
            tmp_file,
            expected_ids=target_ids,
            check_inchikey14=validate_inchikey14,
            output_format=fmt,
        )
        if not is_valid:
            err_msg = f"Submission staging file failed integrity checks:\n" + "\n".join(val_errors[:10])
            raise SubmissionValidationError(err_msg)

        # In-memory DataFrame validation of staged file
        staged_df = pd.read_csv(tmp_file)
        try:
            validate_submission(staged_df, expected_ids=target_ids, output_format=fmt)
        except AssertionError as exc:
            raise SubmissionValidationError(f"Submission DataFrame failed integrity check: {exc}") from exc

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
