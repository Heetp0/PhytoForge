import pytest
from src.retrieval.transductive_networking import (
    BOTANICAL_AGLYCONE_LIBRARY,
    TransductiveMolecularNetwork,
)
from src.pipeline import CASMIOmegaPipeline
import pandas as pd
import numpy as np


def test_botanical_aglycone_library_content():
    """Verify botanical aglycone library has >= 20 verified plant aglycones with valid properties."""
    assert len(BOTANICAL_AGLYCONE_LIBRARY) >= 20
    for name, data in BOTANICAL_AGLYCONE_LIBRARY.items():
        assert "smiles" in data
        assert "inchikey14" in data
        assert "neutral_mass" in data
        assert len(data["inchikey14"]) == 14
        assert data["neutral_mass"] > 100.0


def test_transductive_network_pre_seeded():
    """Test that TransductiveMolecularNetwork can pre-seed botanical aglycones."""
    net = TransductiveMolecularNetwork(cosine_threshold=0.6)
    net.load_botanical_aglycones()
    assert len(net.spectra) >= 20

    # Quercetin (MW 302.0427) + Hexose (162.0528) -> Quercetin glucoside / Isoquercitrin (MW 464.0955)
    # Add a mock spectrum for Quercetin glucoside
    glucoside_mass = 302.0427 + 162.0528
    dummy_emb = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
    dummy_peaks = np.array([[302.0427, 100.0], [152.0110, 50.0], [136.0160, 40.0]])
    net.add_spectrum("MOCK_GLUCOSIDE", glucoside_mass, dummy_emb, dummy_peaks)

    # Propagate from pre-seeded aglycones
    known = {k: (v["smiles"], v["inchikey14"]) for k, v in BOTANICAL_AGLYCONE_LIBRARY.items()}
    cands = net.propagate_scaffold("MOCK_GLUCOSIDE", known_scaffolds=known, min_shared_peaks=1)
    
    # Should find Quercetin as the aglycone source
    assert len(cands) >= 1
    transformations = [c.transformation for c in cands]
    assert "+Hexose" in transformations


def test_pipeline_pre_seeds_aglycones():
    """Test that CASMIOmegaPipeline initializes with pre-seeded aglycones in solved_scaffolds."""
    pipe = CASMIOmegaPipeline(db_path=None)
    assert len(pipe.solved_scaffolds) >= 20
    assert "REF_QUERCETIN" in pipe.solved_scaffolds or any("QUERCETIN" in k for k in pipe.solved_scaffolds)
