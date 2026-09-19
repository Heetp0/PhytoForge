# Original User Request

## Initial Request — 2026-09-18T16:08:53Z

Build a high-performance, modular machine learning pipeline for the Enveda CASMI 2026 Kaggle Competition (LC-MS/MS to SMILES), combining SOTA models (DreaMS, MIST-CF, MS-GPT) with high-leverage unorthodox strategies (BRICS/Knapsack substructure assembly, transductive test networking, and MMR slot portfolio optimization) evaluated against Kaggle's MRR@25 on InChIKey14.

Working directory: d:/The core/Workspace/CodeSpace/enveda-casmi-2026
Integrity mode: development

## Requirements

### R1. Tri-Tier Core Retrieval Pipeline
Implement the foundational 3-tier candidate generation architecture:
- **Tier 1 (Library Matcher):** Mass-gated spectral vector similarity engine using pre-trained DreaMS/MS2DeepScore embeddings.
- **Tier 2 (Database Ranker):** MIST-CF chemical formula estimation and MIST 4,096-bit fingerprint Tanimoto scoring over a pre-indexed COCONUT/PubChem candidate subset.
- **Tier 3 (De Novo Generator):** MS-GPT / ChemFormer generative decoder for novel chemical scaffolds.

### R2. Unorthodox & High-Leverage Competitive Engines
Implement advanced differentiator modules:
- **Substructure Knapsack Assembler (BRICS + ILP):** Fragment extraction from the 275k training set, solving an integer linear programming knapsack problem over observed MS2 fragment masses to generate 100% synthetically valid candidate graphs.
- **Transductive Test Networking:** Unsupervised GNPS-style spectral networking across unlabeled test spectra to propagate high-confidence identifications to related analogs (glycosylations, methylations, hydroxylations).
- **MMR Slot Portfolio Optimizer:** Rank-decayed Maximum Marginal Relevance (MMR) over the 25 output slots to balance top-candidate confidence with scaffold diversity, preventing cluster collapse and maximizing MRR@25 expectation.

### R3. Chemistry Foundation & Kaggle Submission Safety
Build robust chemical preprocessing and inference execution guardrails:
- Precursor mass de-adducting supporting all 10 competition adducts ([M+H]+, [M+NH4]+, [M+Na]+, [M-H]-, etc.).
- False precursor picking correction (handling +1$ {13}\text{C}$ offsets).
- RDKit tautomer canonicalization and InChIKey14 deduplication across the 25 candidate slots.
- Adaptive runtime throttling ensuring overall inference completes in $< 8\text{ hours}$ (avg $< 5.76\text{ s}$ per test molecule) in an isolated, zero-internet environment.
- Atomic CSV writer guaranteeing valid submission.csv format (exactly 25 semicolon-delimited SMILES, no missing rows).

## Acceptance Criteria

### Verification & Performance
- [ ] Automated unit test suite passes for all 10 adduct calculations and InChIKey14 canonicalization.
- [ ] Tier 1 vector search scores query spectra in $< 5\text{ ms}$ per instance on mock benchmark data.
- [ ] Bitpacked 2048/4096-bit fingerprint Tanimoto calculation scores 10,000 candidates in $< 2\text{ ms}$.
- [ ] BRICS Knapsack solver generates chemically valid, charge-neutral candidate structures matching target monoisotopic mass within $\pm 5\text{ ppm}$.
- [ ] MMR Slot Optimizer ensures all 25 output slots contain distinct InChIKey14 connectivity skeletons with controlled scaffold diversity.
- [ ] End-to-end dry run on sample submission data executes without error and generates a strictly compliant submission.csv.
