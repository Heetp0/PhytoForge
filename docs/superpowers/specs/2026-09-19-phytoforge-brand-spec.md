# Design specification: PhytoForge brand architecture and system identity

## Purpose
This specification documents the brand identity, technical nomenclature, package structure, and positioning for the Enveda CASMI 2026 solution.

## Architecture and naming hierarchy

### 1. Naming conventions
- Python package root: `src` (aliased/packaged as `phytoforge`)
- Command line interface: `python -m src.submission.pipeline` (and `phytoforge`)
- Repository root: `enveda-phytoforge`
- Benchmark paper and competition release: `PhytoForge-14`

### 2. Semantic rationale
- `Phyto`: Anchors the pipeline to plant natural product chemistry, specifically secondary metabolites such as flavonoids, terpenoids, and alkaloids. This reflects the transductive botanical network module that propagates structural annotations across exact neutral mass shifts (+132.0423 Da for pentosyl, +162.0528 Da for hexosyl, and +146.0579 Da for rhamnosyl modifications).
- `Forge`: Reflects the constructive synthesis of molecular candidates. Rather than relying solely on static database matches, the pipeline combines multi-modal retrieval with bounded de novo autoregressive generation and formula-sliced structure search.
- `-14`: Specifies the 14-character planar InChIKey skeleton evaluated by the CASMI 2026 competition. The re-ranker and 25-slot portfolio optimizer target this exact metric space by collapsing stereocenters and tautomers into discrete skeletal connectivity classes using RDKit tautomer canonicalization.

## System boundaries and components

### Component mapping
1. Preprocessing and deconvolution: `src/chemistry/adducts.py` and `src/data/loader.py`
   - Adduct filtering requiring oxygen count $O \ge 1$ for single water loss and $O \ge 2$ for double water loss.
   - Multi-isotopologue deconvolution checking $^{13}\text{C}$ isotopic mass spacing ($\Delta m = 1.003355$ Da).
2. Representation and retrieval:
   - `src/data/multi_energy_fusion.py`: Late attention-weighted pooling of 1024-D DreaMS embeddings across stepped collision energy frames (20, 35, and 50 eV).
   - `src/retrieval/dreams_retrieval.py`: Instrument-stratified cosine retrieval over 1024-D DreaMS embeddings.
   - `src/retrieval/mist_formula_router.py`: Top-3 chemical formula posterior estimation with Shannon entropy gating and plus/minus 1H, plus/minus 1O envelope expansion.
   - `src/retrieval/database_search.py`: Vectorized Morgan fingerprint Tanimoto querying.
   - `src/retrieval/generative_denovo.py`: Bounded autoregressive de novo sampling with a 10-second timeout ceiling.
3. Transductive network: `src/retrieval/transductive_networking.py`
   - Biosynthetic mass-shift propagation across test set spectra with MS/MS fragment ion validation.
4. Meta-ranking and slot allocation:
   - `src/reranking/meta_ranker.py`: 32-feature LambdaMART model trained on cross-library splits.
   - `src/reranking/slot_optimizer.py`: Decision-theoretic expected MRR@25 portfolio optimizer allocating exactly 25 distinct InChIKey14 slots.
5. Execution and submission:
   - `src/submission/runtime_governor.py`: Dynamic 21.3 seconds per spectrum budget governor for 9-hour compliance.
   - `src/submission/writer.py`: Submission writer with atomic filesystem replacement and strict invariant validation (exactly 25 unique alphanumeric keys per molecule, semicolon delimiters, no empty values).
   - `src/submission/validator.py`: CLI submission integrity checker.
   - `src/pipeline.py`: Unified pipeline orchestrator exposing `PhytoForgePipeline` and `CASMIOmegaPipeline`.

## User interaction and CLI definition

### Commands
- Full pipeline run: `python -m src.submission.pipeline --input data/test.parquet --output submissions/submission.csv`
- Format validation: `python -m src.submission.validator --submission submissions/submission.csv --expected-ids data/test_ids.txt`

### Arguments
- `--input`: Path to input mass spectra file (.parquet or .mgf).
- `--output`: Path to output CSV file.
- `--governor`: Maximum execution time per spectrum (default: 21.3s).
- `--vram-cap`: Hardware memory limit (default: 16gb).

## Verification
- Unit test suite: 16 test modules covering all 12 pipeline acceptance gates with 256 passing assertions.
- Metric target: MRR@25 on RDKit tautomer-canonicalized InChIKey14.
