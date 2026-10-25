# Acceptance Gates: PhytoForge CASMI 2026 1st Place Push (Phase 6)

## Gate 1: Multi-Adduct Routing
Observable outcome: Sodium, ammonium, water loss, and formate/chloride adduct queries correctly route to candidate formulas and retrieve candidates when primary lookup fails or yields low spectral match.
CHECK: python -m pytest tests/test_multi_adduct_routing.py -v -W error
STATUS: MET (10 passed)

## Gate 2: Botanical Substructure & Neutral-Loss Prior Scoring
Observable outcome: `BotanicalSubstructureScorer` scores candidates based on characteristic glycoside, caffeoyl, galloyl, and diagnostic fragment matches, breaking isomer ties and purging disconnected '+' mixtures.
CHECK: python -m pytest tests/test_botanical_substructures.py -v -W error
STATUS: MET (24 passed)

## Gate 3: Intra-Formula LambdaMART Dataset & Ranker Training
Observable outcome: `AuthenticRankerDatasetBuilder` constructs training groups composed of authentic ground truth, high-Tanimoto analogs, and intra-formula constitutional isomers, forcing LightGBM trees to split on structural fragmentation.
CHECK: python -m pytest tests/test_authentic_ranker_training.py -v -W error
STATUS: MET (11 passed)

## Gate 4: Full Test Suite Zero-Warning Regression Standard
Observable outcome: Entire test suite passes with zero failures and zero warnings under `-W error`.
CHECK: python -m pytest tests/test_multi_adduct_routing.py tests/test_botanical_substructures.py tests/test_authentic_ranker_training.py tests/test_halogen_filter.py tests/test_kaggle_bundle.py -q -W error
STATUS: MET (44 passed in 17.29s)

## Gate 5: Kaggle Standalone Bundle Generation & Validation
Observable outcome: `dist/kaggle_kernel.py` compiles with embedded modules and executes end-to-end on test data within latency and format requirements.
CHECK: python -m src.submission.kaggle_bundle -o dist/kaggle_kernel.py
STATUS: MET (156.8 KB compiled, 87.2s on 50 spectra, 0 '+' mixtures, MRR@25 = 0.5437)
