from pathlib import Path
import sqlite3
import tempfile
from typing import List
import numpy as np
import pandas as pd
import pytest

from src.pipeline import CASMIOmegaPipeline
from src.retrieval.database_search import DBCandidate
from src.retrieval.mist_formula_router import ExactMassFormulaIndex


def test_multi_adduct_secondary_formula_lookup():
    """Verify that a sodium adduct [M+Na]+ at precursor m/z 309.0369 recovers C15H10O6 (neutral mass 286.0477 Da)."""
    index = ExactMassFormulaIndex(formulas=["C15H10O6", "C16H12O7"])

    # Precursor for [M+Na]+ where neutral = 286.0477 + 22.989218 = 309.0369
    precursor_mz = 309.0369

    # Primary lookup as [M+H]+ (309.0369 - 1.0073 = 308.0296) does NOT match C15H10O6 (286.0477)
    primary_neutral = precursor_mz - 1.007276
    primary_results = index.find_formulas_by_mass(target_mass=primary_neutral, tolerance_ppm=25.0)
    primary_forms = [f for f, _ in primary_results]
    assert "C15H10O6" not in primary_forms

    # Secondary hypothesis for [M+Na]+: neutral = precursor - 22.989218 = 286.0477
    na_neutral = precursor_mz - 22.989218
    na_results = index.find_formulas_by_mass(target_mass=na_neutral, tolerance_ppm=25.0)
    na_forms = [f for f, _ in na_results]
    assert "C15H10O6" in na_forms


def test_multi_adduct_ammonium_and_water_loss():
    """Verify ammonium adduct and water-loss in silico mass recovery."""
    index = ExactMassFormulaIndex(formulas=["C15H10O5", "C21H20O11"])
    # C15H10O5 neutral mass = 270.0528 Da
    # [M+NH4]+ precursor = 270.0528 + 18.033823 = 288.0866 Da
    nh4_precursor = 288.0866
    nh4_neutral = nh4_precursor - 18.033823
    nh4_results = index.find_formulas_by_mass(target_mass=nh4_neutral, tolerance_ppm=25.0)
    assert "C15H10O5" in [f for f, _ in nh4_results]

    # [M-H2O+H]+ precursor = 270.0528 - 18.010565 + 1.007276 = 253.0495 Da
    loss_precursor = 253.0495
    recovered_neutral = loss_precursor + 18.010565 - 1.007276
    loss_results = index.find_formulas_by_mass(target_mass=recovered_neutral, tolerance_ppm=25.0)
    assert "C15H10O5" in [f for f, _ in loss_results]


def test_multi_adduct_fallback_condition_predicate():
    """Test the exact multi-adduct fallback routing condition."""
    # Condition: (len(track2_cands) < 5) or (len > 0 and max(score) < 0.20)
    # Case 1: Sparse candidates (< 5) -> True
    sparse_cands = [
        DBCandidate(smiles="C1", inchikey14="IK01", tanimoto_score=0.9),
        DBCandidate(smiles="C2", inchikey14="IK02", tanimoto_score=0.8),
    ]
    needs_fallback_sparse = (len(sparse_cands) < 5) or (
        len(sparse_cands) > 0 and max((float(getattr(c, "tanimoto_score", 0.0)) for c in sparse_cands), default=0.0) < 0.20
    )
    assert needs_fallback_sparse is True

    # Case 2: Sufficient count (>= 5) but all low scoring (< 0.20) -> True
    low_score_cands = [
        DBCandidate(smiles=f"C{i}", inchikey14=f"IK{i:02d}", tanimoto_score=0.01)
        for i in range(10)
    ]
    needs_fallback_low = (len(low_score_cands) < 5) or (
        len(low_score_cands) > 0 and max((float(getattr(c, "tanimoto_score", 0.0)) for c in low_score_cands), default=0.0) < 0.20
    )
    assert needs_fallback_low is True

    # Case 3: Sufficient count (>= 5) with high match score (>= 0.20) -> False
    good_cands = [
        DBCandidate(smiles=f"C{i}", inchikey14=f"IK{i:02d}", tanimoto_score=0.85)
        for i in range(6)
    ]
    needs_fallback_good = (len(good_cands) < 5) or (
        len(good_cands) > 0 and max((float(getattr(c, "tanimoto_score", 0.0)) for c in good_cands), default=0.0) < 0.20
    )
    assert needs_fallback_good is False


def test_multi_adduct_pipeline_sodium_recovery(tmp_path: Path):
    """End-to-end pipeline test: recover C15H10O6 (IK14DBFLAV0002) via [M+Na]+ routing."""
    db_file = tmp_path / "test_compounds.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE candidates ("
        "formula TEXT, inchikey14 TEXT, smiles TEXT, fingerprint BLOB, "
        "PRIMARY KEY (formula, inchikey14)) WITHOUT ROWID;"
    )
    conn.execute("CREATE TABLE _schema_metadata (key TEXT PRIMARY KEY, value TEXT);")
    conn.execute("INSERT INTO _schema_metadata VALUES (?, ?)", ("schema_version", "1.0"))
    fp_blob = np.zeros(32, dtype=np.uint64).tobytes()
    conn.execute(
        "INSERT INTO candidates VALUES (?, ?, ?, ?)",
        ("C15H10O6", "IK14DBFLAV0002", "c1cc(O)c2c(=O)cc(-c3ccc(O)cc3)oc2c1", fp_blob),
    )
    conn.commit()
    conn.close()

    pipeline = CASMIOmegaPipeline(db_path=db_file)
    test_df = pd.DataFrame({
        "id": ["SPEC_NA"],
        "precursor_mz": [309.0369],  # [M+Na]+ for C15H10O6
        "polarity": ["positive"],
    })
    out_csv = tmp_path / "submission.csv"
    pipeline.run(test_df, output_path=out_csv)

    aggregated_iks = [c["inchikey14"] for c in pipeline.last_aggregated_cands]
    assert "IK14DBFLAV0002" in aggregated_iks


def test_multi_adduct_pipeline_negative_formate_recovery(tmp_path: Path):
    """End-to-end pipeline test: recover C15H10O5 (IK14DBFLAV0001) via [M+HCOO]- routing."""
    db_file = tmp_path / "test_compounds_neg.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE candidates ("
        "formula TEXT, inchikey14 TEXT, smiles TEXT, fingerprint BLOB, "
        "PRIMARY KEY (formula, inchikey14)) WITHOUT ROWID;"
    )
    conn.execute("CREATE TABLE _schema_metadata (key TEXT PRIMARY KEY, value TEXT);")
    conn.execute("INSERT INTO _schema_metadata VALUES (?, ?)", ("schema_version", "1.0"))
    fp_blob = np.zeros(32, dtype=np.uint64).tobytes()
    conn.execute(
        "INSERT INTO candidates VALUES (?, ?, ?, ?)",
        ("C15H10O5", "IK14DBFLAV0001", "c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1", fp_blob),
    )
    conn.commit()
    conn.close()

    pipeline = CASMIOmegaPipeline(db_path=db_file)
    test_df = pd.DataFrame({
        "id": ["SPEC_NEG_FORMATE"],
        "precursor_mz": [315.0510],  # [M+HCOO]- for C15H10O5 (270.0528 + 44.9982)
        "polarity": ["negative"],
    })
    out_csv = tmp_path / "submission.csv"
    pipeline.run(test_df, output_path=out_csv)

    aggregated_iks = [c["inchikey14"] for c in pipeline.last_aggregated_cands]
    assert "IK14DBFLAV0001" in aggregated_iks
