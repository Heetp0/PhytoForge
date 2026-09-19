# E2E Test Infra: Enveda CASMI 2026 Pipeline

## Test Philosophy
- Opaque-box, requirement-driven: Derived strictly from `ORIGINAL_REQUEST.md` and user-facing specifications without coupling to internal implementation classes.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.
- Progressive testability: Tests run from unit components through integration up to full end-to-end dry run.

## Feature Inventory & Test Matrix
| # | Feature | Requirement Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Scenario) |
|---|---------|-------------------|:----------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 | 10 Adduct De-adducting | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 2 | False Precursor $^{13}\text{C}$ Correction | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 3 | Tautomer Canonicalization | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 4 | InChIKey14 Deduplication | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 5 | Atomic CSV Submission Writer | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 6 | Adaptive Runtime Governor | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 7 | Tier 1 Mass-Gated Vector Search (<5ms) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 8 | Tier 2 Bitpacked Tanimoto Ranker (<2ms) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 9 | Tier 3 De Novo Generative Decoder | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 10 | BRICS Knapsack Assembler ($\pm 5\text{ ppm}$) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 11 | Transductive Test Networking | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 12 | MMR Slot Portfolio Optimizer (25 slots) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 13 | Full Pipeline Dry Run (submission.csv) | ORIGINAL_REQUEST §Acceptance | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Runner**: PyTest (`pytest -v tests/`)
- **Directory Layout**:
  ```
  tests/
  ├── conftest.py                   # Shared fixtures, synthetic test spectra, molecules
  ├── test_tier1_features.py        # Tier 1: Isolated feature coverage (>=5 per feature)
  ├── test_tier2_boundaries.py      # Tier 2: Boundary value analysis, invalid inputs, edge cases
  ├── test_tier3_combinations.py    # Tier 3: Pairwise interactions between modules
  └── test_tier4_scenarios.py       # Tier 4: Real-world workflow execution & submission validation
  ```
- **Pass/Fail Semantics**:
  - Exit code 0.
  - Strict performance thresholds asserted:
    - Tier 1 query scoring $< 5.0\text{ ms}$.
    - Tier 2 10k 4096-bit Tanimoto scoring $< 2.0\text{ ms}$.
    - Knapsack error $\le 5.0\text{ ppm}$.
    - Submission CSV validation: exactly 25 semicolon-delimited SMILES per molecule, 100% valid InChIKeys.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity | Target Behavior |
|---|----------|--------------------|------------|-----------------|
| 1 | Standard High-Confidence Library Hit | Adducts, Tier 1, MMR, Writer | Low | Precursor matches library hit, rank 1 locked, valid CSV output |
| 2 | Novel Natural Product with Mass Shift | Adducts, Networking, Knapsack, MMR | High | Unseen molecule linked to analog via glycosyl shift (+162.05 Da), knapsack solves fragments |
| 3 | Mispicked $^{13}\text{C}$ Precursor | $^{13}\text{C}$ correction, Tier 2, MMR | Medium | Precursor $m/z$ corrected by $-1.003355$ Da, candidate matched in DB |
| 4 | Degraded / Low-Confidence Query | Tier 1, Tier 2, Tier 3, Knapsack, MMR | High | Fallback through Tier 3 and Knapsack, MMR diversifies across 25 slots |
| 5 | Full Multi-Molecule Competition Batch | All features + Runtime Governor + Writer | High | End-to-end dry run over batch, completes $<5.76\text{s/mol}$, atomic submission generated |

## Minimum Thresholds
- Tier 1: $\ge 65$ test cases ($5 \times 13$)
- Tier 2: $\ge 65$ test cases ($5 \times 13$)
- Tier 3: $\ge 15$ test cases (major pairwise integrations)
- Tier 4: $\ge 5$ end-to-end application scenarios
- **Total Suite Minimum**: $\ge 150$ verifiable test assertions
