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

- [ ] G10: Module 6 Submission Integrity Validator rejects malformed rows, duplicate InChIKey14s, or NaNs, and runtime governor aligns to 21.3s/spectrum
  CHECK: python -m pytest tests/test_submission_governor.py -v
  EXPECT: 2 passed
  EVIDENCE: pending

- [ ] G11: Unified End-to-End Pipeline integration executes from raw query to verified submission.csv with real module wiring and dynamic runtime governor
  CHECK: python -m pytest tests/test_e2e_pipeline.py -v
  EXPECT: 1 passed
  EVIDENCE: pending

- [ ] G12: Full Project Test Suite passes with zero regressions
  CHECK: python -m pytest tests/ -q
  EXPECT: passed
  EVIDENCE: pending
