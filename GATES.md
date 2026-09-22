# Gates: CASMI-Omega v2 Pipeline

OWNS: src/**, tests/**, configs/**, submissions/**

Scope: End-to-end implementation and verification of the CASMI-Omega v2 molecular identification pipeline for Kaggle Enveda CASMI 2026.

- [x] G1: Chemistry Physics Preprocessor passes physical plausibility (O >= 2 for [M-2H2O+H]+) and 13C multi-isotopologue deconvolution
  CHECK: python -m pytest tests/test_chemistry_physics.py -v
  EXPECT: 3 passed
  EVIDENCE: Passed in commit 32777f8 (3 passed in 0.29s, 0 regressions in test_m1_chemistry_submission.py)


- [x] G2: Multi-Energy Spectral Fusion Engine correctly fuses per-collision energy DreaMS embeddings with attention weighting
  CHECK: python -m pytest tests/test_multi_energy_fusion.py -v
  EXPECT: 9 passed
  EVIDENCE: Passed in commit a6f0132 (9 passed in 0.24s, 0 regressions in test_m1_chemistry_submission.py and test_chemistry_physics.py)

- [x] G3: MIST-CF Soft Formula Router expands top-3 formula posteriors (+-1H, +-1O) and triggers high-entropy fallback
  CHECK: python -m pytest tests/test_mist_formula_router.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit cffb9c7 (5 passed in 0.07s, 0 regressions in test_chemistry_physics.py, test_multi_energy_fusion.py, test_m1_chemistry_submission.py)

- [x] G4: Track 1 Calibrated DreaMS Library Retrieval retrieves with instrument-stratified thresholds and FP16 re-scoring
  CHECK: python -m pytest tests/test_dreams_retrieval.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit 79b8cd6 (6 passed in 0.08s, 0 regressions across 68 tests in test_chemistry_physics.py, test_multi_energy_fusion.py, test_mist_formula_router.py, test_m1_chemistry_submission.py)

- [x] G5: Track 2 Soft Database Search queries formula batches with vectorized Tanimoto scoring and NP diversity fallback
  CHECK: python -m pytest tests/test_database_search.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit 6c30bdc (6 passed in 0.05s, 0 regressions across 74 tests in test_chemistry_physics.py, test_multi_energy_fusion.py, test_mist_formula_router.py, test_dreams_retrieval.py, test_m1_chemistry_submission.py)

- [x] G6: Track 3 Generative De Novo Sampler enforces non-blocking execution timeout and native generation step limits
  CHECK: python -m pytest tests/test_generative_denovo.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit 7cd16bd (6 passed in 0.25s, 0 regressions across 80 tests in regression suite)

- [x] G7: Module 3 Neutral-Mass Transductive Network propagates solved scaffolds across expanded delta library (+pentose, +hexose, +rhamnose, +glucuronide, +malonyl) with >= 2 fragment co-validation peaks
  CHECK: python -m pytest tests/test_transductive_networking.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit 1710298 (6 passed in 0.12s, 0 regressions across 86 tests in regression suite)

- [x] G8: Module 4 30+ Feature GBDT Meta-Ranker extracts >= 30 multimodal features across spectral, chemical, and cross-track agreement
  CHECK: python -m pytest tests/test_meta_ranker.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit 7017a37 (5 passed in 0.08s, 0 regressions across 92 tests in regression suite)

- [x] G9: Module 5 Planar InChIKey14 Decision-Theoretic Slot Optimizer guarantees exactly 25 unique strictly alphanumeric planar skeletons
  CHECK: python -m pytest tests/test_slot_optimizer.py -v
  EXPECT: 1 passed
  EVIDENCE: Passed in commit 077867c (6 passed in 0.08s, 0 regressions across 97 tests in regression suite)

- [x] G10: Module 6 Submission Integrity Validator rejects malformed rows, duplicate InChIKey14s, or NaNs, and runtime governor aligns to 21.3s/spectrum
  CHECK: python -m pytest tests/test_submission_governor.py -v
  EXPECT: 2 passed
  EVIDENCE: Passed in commit 215b81a (4 passed in 0.15s, 0 regressions across 103 tests in regression suite)

- [x] G11: Unified End-to-End Pipeline integration executes from raw query to verified submission.csv with real module wiring and dynamic runtime governor
  CHECK: python -m pytest tests/test_e2e_pipeline.py -v
  EXPECT: 1 passed
  EVIDENCE: Passed in commit bcfc0e0 (4 passed in 0.36s, 0 regressions across 111 tests in regression suite)

- [x] G12: Full Project Test Suite passes with zero regressions
  CHECK: python -m pytest tests/ -q
  EXPECT: passed
  EVIDENCE: Passed (258 passed, 0 warnings, 0 failures across all 16 test modules in 10.14s with -W error)

- [x] G13: Comprehensive Testing Overhaul (Multi-Agent TDD Suite: 540 tests)
  CHECK: python -m pytest tests/ -v -W error
  EXPECT: 540 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (540 passed, 0 warnings, 0 failures across 25 test modules in 25.61s with -W error)

- [x] G14: Dual-Format Submission & Pipeline Harmonization (582 tests)
  CHECK: python -m pytest tests/test_dual_format_submission.py -v -W error
  EXPECT: 42 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (42 passed in test_dual_format_submission.py; 582 passed in 64.48s across full suite with 0 warnings)

- [x] G15: Offline Asset Indexing & Precomputation Engine (Phase 2: R1-R5)
  CHECK: python -m pytest tests/test_db_indexer.py tests/test_spectral_indexer.py tests/test_botanical_knowledge.py tests/test_indexer_cli.py tests/test_adversarial_db_indexer.py tests/test_adversarial_spectral_indexer.py -v -W error
  EXPECT: 157 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (111 feature tests + 46 adversarial tests passed under -W error; Tanimoto popcount benchmark verified at 0.27 ms < 2.0 ms per 10,000 candidates; 739 total regression tests passing cleanly)

- [x] G16: Phase 3 Fragment Library & BRICS Knapsack Assembler (R1-R3)
  CHECK: python -m pytest tests/test_brics_knapsack.py tests/test_fragment_library.py -v -W error
  EXPECT: 31 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (31 passed in 0.58s with -W error; verified dummy atom stripping, neutral loss library int-keys, and knapsack assembler meet-in-the-middle)

- [x] G17: Phase 3 Pipeline Context Manager & End-to-End Integration (R5-R6)
  CHECK: python -m pytest tests/test_pipeline_phase3.py -v -W error
  EXPECT: 13 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (13 passed in 12.22s with -W error; verified pipeline context manager, db_path dependency injection, Track K knapsack candidate aggregation, 5-spectrum E2E dry-run in <15s, and dual output schemas)

- [x] G18: Phase 3 Full Repository Test Suite & Adversarial Stress Verification
  CHECK: python -m pytest tests/ -q -W error
  EXPECT: 818 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (818 passed, 0 warnings, 0 failures in 248.41s across all 37 test modules with -W error; full regression and adversarial suite clean, 0 timeouts, strict governor compliance)

- [x] G19: Phase 4 Evaluation Metrics & Bemis-Murcko Scaffold Splitter (R1)
  CHECK: python -m pytest tests/test_evaluation_metrics.py -v -W error
  EXPECT: 36 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (36 passed in 1.11s with -W error; verified MRR@k, Top-K accuracies, and zero-leakage Bemis-Murcko scaffold cross-validation with acyclic carbon-chain fallback)

- [x] G20: Phase 4 GBDT LambdaMART Ranker Training & Native Booster Engine (R2, R3)
  CHECK: python -m pytest tests/test_ranker_training.py -v -W error
  EXPECT: 16 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (16 passed in 4.93s with -W error; verified synthetic listwise dataset generation, LightGBM LambdaMART ranker training, native .txt model persistence, and vectorized batch prediction with zero regression on heuristic fallback)

- [x] G21: Phase 4 Kaggle Bundle Generator & Cross-Validation Benchmark CLI (R4, R5)
  CHECK: python -m pytest tests/test_kaggle_bundle.py tests/test_evaluation_benchmark.py tests/test_adversarial_phase4_bundle_benchmark.py -v -W error
  EXPECT: 39 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (39 passed in 19.82s with -W error; verified Kaggle environment auto-detection, standalone bundle compilation to dist/kaggle_kernel.py, and end-to-end latency benchmark CLI under governor limits)

- [x] G22: Phase 4 Full Repository Test Suite Verification (All 909 Tests)
  CHECK: python -m pytest tests/ -q -W error
  EXPECT: 909 passed, 0 warnings, 0 failures
  EVIDENCE: Passed (909 passed, 0 warnings, 0 failures in 290.44s across all 42 test modules with -W error; full regression and adversarial suite clean, 0 timeouts, strict governor compliance)

