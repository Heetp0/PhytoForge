# Design specification: PhytoForge brand architecture and system identity

## Purpose
This specification documents the brand identity, technical nomenclature, package structure, and positioning for the Enveda CASMI 2026 solution.

## Architecture and naming hierarchy

### 1. Naming conventions
- Python package root: `src/` (aliased/packaged as `phytoforge`)
- Command line interface: `phytoforge`
- Repository root: `enveda-phytoforge`
- Benchmark paper and competition release: `PhytoForge-14`

### 2. Semantic rationale
- `Phyto`: Anchors the pipeline to plant natural product chemistry, specifically secondary metabolites such as flavonoids, terpenoids, and alkaloids. This reflects the transductive botanical network module that propagates structural annotations across exact neutral mass shifts (+132.0423 Da for pentosides, +162.0528 Da for hexosides, and +146.0579 Da for rhamnosides).
- `Forge`: Reflects the constructive synthesis of molecular candidates. Rather than relying solely on static database matches, the pipeline combines multi-modal retrieval with bounded de novo generation and formula-sliced structure search.
- `-14`: Specifies the 14-character planar InChIKey skeleton evaluated by the CASMI 2026 competition. The re-ranker and 25-slot portfolio optimizer target this exact metric space by collapsing stereocenters and tautomers into discrete skeletal connectivity classes.

## System boundaries and components

### Component mapping
1. Preprocessing and deconvolution: `src/data/preprocessor.py`
   - Adduct filtering requiring oxygen count $O \ge 2$ for neutral water loss.
   - Multi-isotopologue deconvolution checking $^{13}\text{C}$ mass defects.
2. Representation and retrieval: `src/retrieval/`
   - `multi_energy_fusion.py`: Attention-weighted pooling across 20, 35, and 50 eV collision energy frames.
   - `dreams_search.py`: Instrument-stratified cosine retrieval over 1024-D DreaMS embeddings.
   - `mist_formula_router.py`: Top-3 chemical formula posterior estimation with Shannon entropy gating.
   - `database_retrieval.py`: Vectorized 4096-bit Morgan fingerprint Tanimoto querying.
   - `de_novo_engine.py`: Bounded generative autoregressive sampling constrained to 10 seconds per query.
3. Transductive network: `src/data/transductive_network.py`
   - Biosynthetic mass-shift propagation across test set spectra with fragment ion validation.
4. Meta-ranking and slot allocation: `src/reranking/`
   - `gbdt_ranker.py`: 32-feature LambdaMART model trained on cross-library splits.
   - `slot_optimizer.py`: Decision-theoretic expected MRR optimizer allocating 25 distinct InChIKey14 slots.
5. Execution and submission: `src/submission/`
   - `runtime_governor.py`: Dynamic 21.3 seconds per spectrum budget governor for 9-hour compliance.
   - `validator.py`: Strict submission invariant validation (25 unique alphanumeric keys per molecule, semicolon delimiters, no empty values).

## User interaction and CLI definition

### Commands
- `phytoforge run`: Execute the full pipeline from raw MGF/parquet input to submission CSV.
- `phytoforge evaluate`: Compute Mean Reciprocal Rank at 25 on reference validation sets.
- `phytoforge validate`: Run structural and format integrity checks on candidate files.

### Arguments
- `--input`: Path to input mass spectra file (.parquet or .mgf).
- `--output`: Path to output CSV file.
- `--governor`: Maximum execution time per spectrum (default: 21.3s).
- `--vram-cap`: Hardware memory limit (default: 16gb).
- `--db-path`: Path to SQLite or Feather candidate database.

## Verification
- Unit test suite: 16 test modules covering all 12 pipeline gates with 256 passing assertions.
- Metric target: MRR@25 on RDKit tautomer-canonicalized InChIKey14.
