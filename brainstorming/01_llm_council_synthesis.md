# LLM Council Deliberation & Synthesis Report
**Competition:** Enveda CASMI 2026 (Kaggle) — Molecule ID from Mass Spectra  
**Target:** Maximize MRR@25 on RDKit Tautomer-Canonicalized `InChIKey14`  
**Deliberation Stage:** Stage 1 (First Opinions) -> Stage 2 (Peer Critique & Scoring) -> Stage 3 (Chairman Synthesis)

---

## Executive Summary: The 3 Critical Verdicts of the Council

1. **Fatal Bottleneck Discarded:**  
   The council unanimously and unconditionally rejects online in-silico fragmentation (MetFrag, CFM-ID) and brute-force unindexed cosine spectral searches across the 2.5M training set. Within Kaggle’s 9-hour inference limit (~5.76s per molecule), online fragmentation causes **guaranteed Time Limit Exceeded (TLE)**.
2. **Formula Decomposition Before Structure Search:**  
   Filtering candidates purely by precursor $m/z$ returns thousands of false isomers. The council mandates **precursor mass defect formula decomposition (Seven Golden Rules & Senior's rules)**, pruning the candidate universe by $100\times$ before any fingerprint similarity scoring.
3. **The Core Winning Architecture (Dual-Encoder + Bitpacked Tanimoto):**  
   - **Class 1 (Library):** $\pm 10$ ppm mass slice + DreaMS / MS2DeepScore INT8 vector similarity. Lock into Rank 1 if similarity $>0.82$.
   - **Class 2 (Database):** MIST-CF neural fingerprint predictor (4096-bit composite) scored against offline-indexed COCONUT/PubChem using AVX/CUDA bitwise popcount Tanimoto ($<2$ ms per query).
   - **Class 3 (De Novo):** InChIKey14 diversity-constrained autoregressive beam search (ChemFormer).
   - **Ensemble Re-ranker:** LightGBM LambdaMART ranker using spectral match, NP-likeness prior, and fragment coverage features.
   - **Slot Packing:** Strict InChIKey14 deduplication to ensure all 25 output slots test distinct 2D skeletal frameworks.

---

## Complete Pipeline Blueprint

```
                  Raw Query MS/MS Spectra per molecule_id
                  (Multi-collision energy ramps, adduct modes)
                                      │
                                      ▼
           [Step 1: Precursor De-Adducting & De-Isotoping]
           - Correct M+1 / M+2 false precursor picking
           - Score adduct plausibility ([M+H]+, [M+Na]+, [M-H]-, etc.)
                                      │
                                      ▼
             [Step 2: Exact Formula Attribution (SIRIUS)]
             - Decompose precursor mass defect into valid formulas
             - Sub-formula tree consistency across MS2 fragments
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
  [Track 1: Class 1]            [Track 2: Class 2]            [Track 3: Class 3]
  Mass-filtered (±10 ppm)       MIST-CF predicted 4096-bit    ChemFormer generative
  DreaMS INT8 vector match      fingerprint vs. pre-indexed   SMILES with mass defect
  against 2.5M library          COCONUT/PubChem candidates    constraint
  (If sim > 0.82 -> Rank 1)     via vectorized popcount (<2ms) & InChIKey14 diversity
       │                              │                              │
       └──────────────────────────────┼──────────────────────────────┘
                                      ▼
            [Step 3: Unified Candidate Pool (Top 50-100 Candidates)]
                                      │
                                      ▼
            [Step 4: LightGBM LambdaMART Learning-to-Rank (LTR)]
            - Spectral cosine + Neural fingerprint Tanimoto
            - Precursor mass ppm error + Fragment intensity coverage
            - Natural Product Likeness (NP-score) + COCONUT prior
                                      │
                                      ▼
            [Step 5: InChIKey14 Deduplication & Slot Packing]
            - Enforce RDKit tautomer canonicalization
            - Ensure 25 slots test 25 distinct connectivity skeletons
                                      │
                                      ▼
            [Step 6: submission.csv Generation with Safety Wrappers]
            - Adaptive runtime manager (fast/normal/deep mode)
            - 100% row match guarantee against test set
```
