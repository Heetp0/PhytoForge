# Project: Enveda CASMI 2026 Kaggle Competition Pipeline

## Architecture
Modular, high-performance machine learning and chemical retrieval pipeline for LC-MS/MS to SMILES identification evaluated on MRR@25 of InChIKey14:
- **`src/chemistry/`**: Chemical foundation, 10 adduct de-adducting, isotopic $^{13}\text{C}$ offset correction, RDKit salt stripping, charge neutralization, tautomer canonicalization, and InChIKey14 extraction.
- **`src/retrieval/`**: Tri-Tier candidate retrieval:
  - Tier 1 (Library Matcher): Mass-gated spectral vector similarity with DreaMS embeddings (<5ms per query).
  - Tier 2 (Database Ranker): Formula estimation and Numba-accelerated bitpacked 4096-bit popcount Tanimoto scoring (<2ms per 10k candidates).
  - Tier 3 (De Novo Generator): Generative decoder interface (MS-GPT / ChemFormer) with mass-defect filtering and chemical grammar checks.
- **`src/reranking/`**: Unorthodox competitive engines:
  - Substructure Knapsack Assembler: BRICS fragment extraction and Meet-in-the-Middle dynamic programming / MILP integer knapsack solver ($\pm 5\text{ ppm}$).
  - Transductive Test Networking: Unsupervised GNPS-style modified cosine networking across test set with metabolic analog propagation.
  - MMR Slot Portfolio Optimizer: Rank-decayed Maximum Marginal Relevance ($\lambda = 1.0 \to 0.35$) guaranteeing 25 distinct InChIKey14 candidate slots.
- **`src/submission/`**: Kaggle compliance:
  - Adaptive Runtime Governor: 3-tier runtime throttling (<8h total, avg <5.76s/query).
  - Atomic CSV Writer: Exactly 25 semicolon-delimited SMILES per query ID via atomic temporary file replacement.
- **`src/data/`**: Data loading and realistic synthetic test/mock spectrum generation for benchmarking and dry runs.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 10 Adduct De-adducting | Exact monoisotopic delta masses with electron mass corrections for all 10 adducts | M1 | ORIGINAL_REQUEST §R3 |
| 2 | False Precursor $^{13}\text{C}$ Correction | Correction of $+1.003355\text{ Da}$ quadrupole isolation mispicks | M1 | ORIGINAL_REQUEST §R3 |
| 3 | Tautomer Canonicalization | RDKit `TautomerEnumerator().Canonicalize()` normalizing tautomers | M1 | ORIGINAL_REQUEST §R3 |
| 4 | InChIKey14 Deduplication | Stripping salts/charges, generating 14-char skeleton InChIKey, and filtering duplicates | M1 | ORIGINAL_REQUEST §R3 |
| 5 | Atomic CSV Submission Writer | Compliant writer producing exactly 25 semicolon-delimited SMILES per row via atomic `os.replace` | M1 | ORIGINAL_REQUEST §R3 |
| 6 | Adaptive Runtime Governor | Throttles execution modes (DEEP, STANDARD, FAST) to guarantee completion in $<8\text{h}$ | M1 | ORIGINAL_REQUEST §R3 |
| 7 | Mock Data Generator | Realistic synthetic spectra generator for development, testing, and benchmarks | M1 | Survey Finding |
| 8 | Tier 1 Mass-Gated Vector Search | Mass-gated spectral vector similarity engine scoring query in $<5\text{ms}$ | M2 | ORIGINAL_REQUEST §R1 |
| 9 | Tier 2 Bitpacked Tanimoto Ranker | Numba JIT popcount Tanimoto scoring 10,000 4096-bit fingerprints in $<2\text{ms}$ | M2 | ORIGINAL_REQUEST §R1 |
| 10 | Tier 3 De Novo Generative Decoder | Conditional generative decoder fallback with mass-defect filtering $\le 10\text{ ppm}$ | M2 | ORIGINAL_REQUEST §R1 |
| 11 | BRICS Knapsack Assembler | Meet-in-the-Middle DP / ILP knapsack producing neutral molecules within $\pm 5\text{ ppm}$ | M3 | ORIGINAL_REQUEST §R2 |
| 12 | Transductive Test Networking | GNPS-style cosine spectral network propagating analog identifications | M3 | ORIGINAL_REQUEST §R2 |
| 13 | MMR Slot Portfolio Optimizer | Rank-decayed MMR slot allocator ensuring 25 distinct InChIKey14 scaffolds | M3 | ORIGINAL_REQUEST §R2 |
| 14 | End-to-End Pipeline Integration | Unified pipeline executing data ingestion, retrieval, reranking, and submission | M4 | ORIGINAL_REQUEST §Acceptance |
| 15 | Full E2E Test Suite Validation | 100% pass across Tiers 1-4 opaque-box test suite published by E2E track | M4 | Project Architecture |
| 16 | Adversarial Coverage Hardening | Tier 5 white-box challenger stress testing and coverage hardening | M4 | Project Architecture |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Chemistry Foundation & Submission Safety | `src/chemistry/`, `src/submission/`, `src/data/mock_data.py` | none | PLANNED |
| M2 | Tri-Tier Retrieval Pipeline | `src/retrieval/` (Tiers 1, 2, 3) | M1 | PLANNED |
| M3 | Unorthodox Competitive Engines | `src/reranking/` (Knapsack, Networking, MMR) | M1, M2 | PLANNED |
| M4 | End-to-End Pipeline Integration & Verification | `src/pipeline.py`, E2E test execution, Tier 5 hardening | M1, M2, M3, TEST_READY | PLANNED |

## Interface Contracts

### `src/chemistry/adducts.py` ↔ `src/retrieval/tier1_matcher.py` & `tier2_ranker.py`
- `calculate_neutral_mass(precursor_mz: float, adduct: str) -> float`
- `get_adduct_candidates(precursor_mz: float, mode: str) -> List[Tuple[str, float]]`
- `correct_precursor_mz(observed_mz: float, ms2_peaks: np.ndarray) -> float`

### `src/chemistry/standardizer.py` ↔ `src/reranking/mmr_optimizer.py` & `src/submission/writer.py`
- `standardize_mol(smiles: str) -> Tuple[Optional[str], Optional[str]]`: Returns `(canonical_smiles, inchikey14)`
- `deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]`

### `src/retrieval/` ↔ `src/reranking/`
- Dataclass `Candidate`:
  ```python
  @dataclass
  class Candidate:
      smiles: str
      inchikey14: str
      score: float
      source_tier: str  # 'tier1', 'tier2', 'tier3', 'knapsack', 'network'
      neutral_mass: float
      fingerprint: Optional[np.ndarray] = None  # uint64 bitpacked array
  ```
- Retrieval output: `List[Candidate]` per query spectrum.

### `src/reranking/mmr_optimizer.py` ↔ `src/submission/writer.py`
- `optimize_portfolio(candidates: List[Candidate], top_k: int = 25) -> List[Candidate]`
- Guarantee: Exactly $\le 25$ candidates with pairwise unique `inchikey14`.

### `src/submission/writer.py`
- `write_submission(predictions: Dict[str, List[str]], output_path: Path) -> Path`:
  - `molecule_id,smiles`
  - Semicolon-delimited SMILES (exactly 25 slots)
  - Atomic write via temporary file and `os.replace`.

## Code Layout
```
d:/The core/Workspace/CodeSpace/enveda-casmi-2026/
├── configs/                 # Runtime & model configuration files
├── data/                    # Dataset storage (raw, processed, mock)
├── models/                  # Precomputed indexes, embeddings, weights
├── src/
│   ├── chemistry/           # M1: Adducts, standardizer, 13C correction
│   ├── data/                # M1: Data loaders, synthetic mock generator
│   ├── retrieval/           # M2: Tier 1, Tier 2, Tier 3 engines
│   ├── reranking/           # M3: BRICS knapsack, networking, MMR optimizer
│   ├── submission/          # M1: Runtime governor, atomic CSV writer
│   └── pipeline.py          # M4: End-to-end pipeline runner
├── tests/                   # E2E Test Suite (Tiers 1-5)
├── ORIGINAL_REQUEST.md      # Authoritative specifications
├── PROJECT.md               # Master project index & contracts
├── TEST_INFRA.md            # Master E2E testing specification
└── TEST_READY.md            # Signoff from E2E testing track
```
