import os
import stat
import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.retrieval.mist_formula_router import ExactMassFormulaIndex
from src.data.db_indexer import ChemicalDatabaseRepository


def test_sqlite_readonly_filesystem_simulation():
    """Simulate Kaggle /kaggle/input read-only filesystem permissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_candidates.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE candidates (formula TEXT, inchikey14 TEXT, smiles TEXT, fingerprint BLOB);")
        fp_bytes = b'\x00' * 256
        conn.execute("INSERT INTO candidates VALUES (?, ?, ?, ?);", ('C10H12O2', 'TESTINCHIKEY14', 'CC(=O)Oc1ccccc1', fp_bytes))
        conn.commit()
        conn.close()

        # Mark file as read-only (simulates read-only mount on Linux)
        os.chmod(str(db_path), stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

        try:
            # 1. Test ExactMassFormulaIndex on read-only file
            index = ExactMassFormulaIndex(db_path=db_path)
            assert len(index.formulas) == 1
            assert index.formulas[0] == "C10H12O2"

            # 2. Test ChemicalDatabaseRepository on read-only file
            with ChemicalDatabaseRepository(db_path) as repo:
                meta, fps = repo.search_by_formulas(["C10H12O2"])
                assert len(meta) == 1
                assert meta[0][2] == "CC(=O)Oc1ccccc1"
                assert fps.shape == (1, 32)
        finally:
            # Reset write permission so temporary directory cleanup succeeds
            os.chmod(str(db_path), stat.S_IWRITE | stat.S_IREAD)
