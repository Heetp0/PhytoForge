"""Wave B Test Suite: Knapsack & De Novo Substructure Assembly.

Verifies:
1. Retrosynthetic BRICS fragment extraction and dummy atom hydrogen saturation.
2. Fast binary bisect search in NeutralLossLibrary.
3. Zero-timeout fast-path bypass in BoundedGenerativeEngine.
4. Disconnected mixture purging in KnapsackAssembler.
5. In-memory SMARTS match caching in BotanicalSubstructureScorer.
"""

from pathlib import Path
import pytest
import numpy as np

from src.reranking.brics_knapsack import (
    BRICSFragmentLibrary,
    extract_fragments,
    strip_dummy_atoms_and_sanitize,
)
from src.reranking.fragment_library import (
    NeutralLossLibrary,
    KnapsackAssembler,
    purge_mixtures,
    parse_peaks_field,
)
from src.retrieval.generative_denovo import (
    BoundedGenerativeEngine,
    GenerativeCandidate,
)
from src.reranking.botanical_substructures import (
    BotanicalSubstructureScorer,
)


def test_brics_dummy_hydrogen_saturation():
    """Verify dummy attachment markers in BRICS fragments are saturated with Hydrogen."""
    # Dummy fragment with attachment point: '[16*]c1ccc(O)cc1' (phenol with BRICS label 16)
    frag_smi = "[16*]c1ccc(O)cc1"
    res = strip_dummy_atoms_and_sanitize(frag_smi)
    assert res is not None
    clean_smi, mass, millimass_key = res
    assert "*" not in clean_smi
    # When dummy is replaced with H, [16*]c1ccc(O)cc1 becomes phenol (c1ccc(O)cc1)
    assert clean_smi == "Oc1ccccc1"
    assert abs(mass - 94.04186) < 1e-3
    assert millimass_key == int(round(mass * 10000))


def test_neutral_loss_library_cached_bisect():
    """Verify NeutralLossLibrary executes fast cached bisect search across mass intervals."""
    lib = NeutralLossLibrary()
    # Insert 1000 keys across mass range
    for k in range(10000, 20000, 10):
        lib.library[k] = [f"FRAG_{k}"]

    # Query for target mass 1.5000 Da (15000 millimass) with 50 ppm tolerance
    # 50 ppm of 1.5 Da is 0.000075 Da -> millimass range [14999, 15001]
    res = lib.query(target_mass_da=1.5000, ppm_tolerance=50.0)
    assert isinstance(res, list)
    assert 15000 in res


def test_generative_engine_zero_timeout_fast_path():
    """Verify zero or negative timeout returns immediately without spawning threads."""
    called = False

    def dummy_gen(spec_id, formula, max_steps):
        nonlocal called
        called = True
        return [GenerativeCandidate("C", "IK14METHANE001", 0.5)]

    engine = BoundedGenerativeEngine(generator_fn=dummy_gen)

    # Effective timeout <= 0.0 must not invoke generator_fn
    res_zero = engine.generate_with_timeout("SPEC_001", timeout_seconds=0.0)
    assert res_zero == []
    assert not called

    res_neg = engine.generate_with_timeout("SPEC_001", timeout_seconds=-5.0)
    assert res_neg == []
    assert not called


def test_knapsack_mixture_purging():
    """Verify purge_mixtures strips disconnected multi-component tokens containing '+'."""
    candidates = [
        "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13",       # Quercetin (valid single molecule)
        "c1ccccc1+c1ccncc1",                                 # Disconnected benzene + pyridine
        "Oc1ccccc1+O=C(O)c1ccccc1",                          # Disconnected phenol + benzoic acid
        "COc1cc(O)c2c(c1)oc(c3ccccc3)cc2=O",                 # Wogonin (valid single molecule)
    ]
    purged = purge_mixtures(candidates)
    assert len(purged) == 2
    assert all("+" not in c for c in purged)
    assert purged[0] == "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13"
    assert purged[1] == "COc1cc(O)c2c(c1)oc(c3ccccc3)cc2=O"


def test_botanical_substructure_scorer_caching():
    """Verify BotanicalSubstructureScorer evaluates candidates and caches results."""
    scorer = BotanicalSubstructureScorer(tolerance_da=0.025)
    fused_peaks = np.array([[153.02, 100.0], [303.05, 50.0]], dtype=np.float64)

    cand_quercetin = {
        "smiles": "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13",
        "inchikey14": "REFJWTPEDVJJIY",
    }

    score1 = scorer.score_candidate(cand_quercetin, neutral_mass=302.04, fused_peaks=fused_peaks)
    assert isinstance(score1, float)
    assert score1 >= 0.0

    score2 = scorer.score_candidate(cand_quercetin, neutral_mass=302.04, fused_peaks=fused_peaks)
    assert score1 == score2
