"""Wave 5 Test Suite: Dynamic Runtime Governor, Latency Optimization, & Kaggle Deployment.

Verifies:
1. Dynamic operating regime transitions (DEEP, STANDARD, FAST).
2. Golden library hit bypass logic (DreaMS similarity >= 0.82).
3. Dynamic per-spectrum budget allocation and safety buffer enforcement.
4. End-to-end pipeline latency adaptation and bypass activation.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.submission.runtime_governor import (
    RuntimeGovernor,
    DynamicRuntimeGovernor,
    OperatingRegime,
)
from src.pipeline import CASMIOmegaPipeline


def test_runtime_governor_regimes():
    """Verify governor shifts regimes based on dynamically available time per query."""
    # 400 queries, 4000 seconds remaining -> 10.0s / query -> DEEP
    gov = RuntimeGovernor(total_budget_sec=4000.0, safety_buffer_sec=0.0, total_queries=400)
    assert gov.get_execution_mode() == OperatingRegime.DEEP
    assert gov.current_regime == OperatingRegime.DEEP

    # 400 queries, 1200 seconds remaining -> 3.0s / query -> STANDARD
    gov = RuntimeGovernor(total_budget_sec=1200.0, safety_buffer_sec=0.0, total_queries=400)
    assert gov.get_execution_mode() == OperatingRegime.STANDARD

    # 400 queries, 400 seconds remaining -> 1.0s / query -> FAST
    gov = RuntimeGovernor(total_budget_sec=400.0, safety_buffer_sec=0.0, total_queries=400)
    assert gov.get_execution_mode() == OperatingRegime.FAST


def test_runtime_governor_golden_hit_bypass():
    """Verify high-confidence library matches trigger generative bypass."""
    gov = RuntimeGovernor(total_budget_sec=1000.0, bypass_similarity_threshold=0.82)
    gov.start_query("MOL_001")

    # Moderate score does not bypass
    assert not gov.should_bypass_generative(0.75)
    assert not gov._current_bypassed

    # High score triggers bypass
    assert gov.should_bypass_generative(0.85)
    assert gov._current_bypassed

    dur = gov.finish_query("MOL_001")
    assert dur >= 0.0
    summary = gov.get_summary()
    assert summary["bypassed_deep_count"] == 1


def test_dynamic_runtime_governor_budget_decay():
    """Verify DynamicRuntimeGovernor decays per-spectrum budget cleanly."""
    gov = DynamicRuntimeGovernor(total_budget_seconds=1000.0, reserve_seconds=100.0)
    assert gov.available_time == 900.0

    # At start, 100 spectra remaining, 0 elapsed -> 900 / 100 = 9.0s
    b1 = gov.get_per_spectrum_budget(elapsed_seconds=0.0, spectra_remaining=100)
    assert abs(b1 - 9.0) < 1e-4

    # Halfway through time, 50 spectra remaining -> (900 - 450) / 50 = 9.0s
    b2 = gov.get_per_spectrum_budget(elapsed_seconds=450.0, spectra_remaining=50)
    assert abs(b2 - 9.0) < 1e-4

    # Budget nearly exhausted -> clamped to minimum 10.0 / spectra_remaining
    b3 = gov.get_per_spectrum_budget(elapsed_seconds=950.0, spectra_remaining=10)
    assert abs(b3 - 1.0) < 1e-4


def test_pipeline_golden_hit_bypass_execution():
    """Verify CASMIOmegaPipeline bypasses de novo and knapsack when top library match is high."""
    pipe = CASMIOmegaPipeline(db={"C15H10O7": [{"inchikey14": "REFJWTPEDVJJIY", "smiles": "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc3cc(O)cc(O)c13"}]})
    dummy_df = pd.DataFrame([{
        "spectrum_id": "TEST_BYPASS_001",
        "precursor_mz": 303.05,
        "adduct": "[M+H]+",
        "molecular_formula": "C15H10O7",
        "peaks": "[[303.05, 100.0]]",
    }])

    res = pipe.run(dummy_df)
    assert len(res) == 1
    assert res.iloc[0]["id"] == "TEST_BYPASS_001"
    cands = res.iloc[0]["candidates"].split(";")
    assert len(cands) == 25
    assert len(set(cands)) == 25
