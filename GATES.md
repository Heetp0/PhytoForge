# Acceptance Gates Ledger — Remaining Subsystems Audit & Optimization
Scope: Enveda CASMI 2026 Comprehensive Non-God Node Subsystem Optimization
Pipeline: Wave A (Chemistry & Preprocessing Core) -> Wave B (Knapsack & De Novo Assembly) -> Wave C (Evaluation & Data Tooling)

---

## Gates

### G-WAVE-A-CHEM: Chemistry & Preprocessing Core Verification
Observable outcome: All chemistry standardization, multi-adduct neutral mass calculations, multi-energy spectral fusion, and isotope carbon/halogen detection pass under strict zero-warning mode without regressions.
CHECK: python -m pytest tests/test_wave_a_chemistry_core.py tests/test_m1_chemistry_submission.py tests/test_halogen_filter.py -q -W error
EXPECT: passed

### G-WAVE-B-KNAP: Knapsack & De Novo Substructure Assembly Verification
Observable outcome: Fragment library parsing, BRICS ILP knapsack mass decomposition, neutral loss mining, and latency-bounded de novo sampling execute cleanly with mixture purging and zero disconnected tokens.
CHECK: python -m pytest tests/test_wave_b_knapsack_assembly.py tests/test_botanical_substructures.py -q -W error
EXPECT: passed

### G-WAVE-C-EVAL: Evaluation & Data Pipeline Verification
Observable outcome: Offline CASMI evaluation metrics (MRR@25, Hit@1, Hit@5, Hit@25), train/test splitters, parquet loading, and ranker dataset builders execute with zero errors under `-W error`.
CHECK: python -m pytest tests/test_wave_c_evaluation_data.py tests/test_authentic_ranker_training.py -q -W error
EXPECT: passed

### G-REGRESS-GLOBAL: Global Repository Zero-Regression Test Suite
Observable outcome: Full repository test suite across all 850+ unit and integration tests passes with zero failures and zero warnings.
CHECK: python -m pytest tests/ -q -W error
EXPECT: passed

### G-KERNEL-VERIFY: Standalone Kernel Bundle Recompilation
Observable outcome: Single-file zero-network `dist/kaggle_kernel.py` compiles successfully and passes AST validation.
CHECK: python -m src.submission.kaggle_bundle
EXPECT: Kaggle standalone kernel successfully generated
