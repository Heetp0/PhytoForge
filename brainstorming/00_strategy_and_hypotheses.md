# Enveda CASMI 2026: Brainstorming & Strategic Hypotheses

## 1. Problem Archetype & Core Dynamics
- **Task**: Predict up to 25 ranked SMILES per `molecule_id` given 1-16 LC-MS/MS spectra.
- **Metric**: MRR@25 on `InChIKey14` (2D skeleton only; tautomers & stereocenters ignored).
- **Test Set**: ~400 molecules across 3 hidden novelty classes:
  - **Class 1**: In public reference spectral libraries.
  - **Class 2**: Structure exists in PubChem / COCONUT (no reference spectra).
  - **Class 3**: Completely novel structure (de novo).

---

## 2. Core Strategic Hypotheses

### Hypothesis 1: Clean Neutral Mass Filtering
- **Rationale**: Accurate precursor mass $m/z$ combined with the known adduct (`[M+H]+`, `[M+Na]+`, `[M-H]-`, etc.) allows calculating the neutral monoisotopic mass to $<5$ ppm error.
- **Action**: Narrow down the search universe from millions of molecules to $<500$ candidates strictly matching the precursor mass and possible molecular formulas.

### Hypothesis 2: High-Confidence Library Matching for Class 1
- **Rationale**: Any test molecule present in the training set or public MassBank/GNPS can be matched with near 100% confidence using spectral similarity (e.g. `matchms` cosine, DreaMS embeddings).
- **Action**: Build an indexed lookup table of all ~275k training molecules + MassBank. If a query spectrum yields similarity $>0.85$, lock that molecule into Rank 1.

### Hypothesis 3: Multi-Spectrum Aggregation
- **Rationale**: Single molecules have multiple spectra at different collision energies (e.g., 20 eV, 40 eV, 60 eV). Low energy retains large fragments (core scaffolds), while high energy provides functional group fingerprints.
- **Action**: Merge and align spectra belonging to the same `molecule_id` rather than predicting independently per spectrum.

### Hypothesis 4: InChIKey14 Slot Optimization
- **Rationale**: The metric only rewards the first correct `InChIKey14`. Providing two stereoisomers or tautomers wastes a slot in the top 25.
- **Action**: Always canonicalize candidate SMILES and enforce uniqueness on `InChIKey14` so all 25 guesses test distinct 2D skeletons.

### Hypothesis 5: De Novo Fallback for Class 3
- **Rationale**: Class 3 molecules cannot be found in databases.
- **Action**: Predict Morgan/MACCS fingerprints directly from spectral peak vectors, then decode or retrieve nearest structural analogs.

---

## 3. Experiment Log & Ideas Backlog
| ID | Idea / Experiment | Target Class | Expected Impact | Status |
|---|---|---|---|---|
| EXP-01 | Exact Adduct & Mass Delta Calculator | All | Foundational | Backlog |
| EXP-02 | Training Library Vector Index (Cosine + DreaMS) | Class 1 | High (Guaranteed hits) | Backlog |
| EXP-03 | COCONUT / PubChem Candidate Pool Generation | Class 2 | High (Covers natural products) | Backlog |
| EXP-04 | Candidate Re-ranker via In-Silico Fragment Matching | Class 2 | Medium-High | Backlog |
| EXP-05 | InChIKey14 Deduplication Post-Processor | All | Free MRR boost | Backlog |
