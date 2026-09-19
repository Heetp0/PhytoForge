# The Ideal Bespoke Methodology: "CASMI-Omega v2"
**Project:** Enveda CASMI 2026 (Kaggle) — Molecule ID from Mass Spectra  
**Version:** 2.0 (Post-Council Architecture Revision — Fully Resolved)  
**Design Principle:** Synthesizing Deep Learning Foundation Models, Chemical Domain Physics, and Unorthodox Transductive/Portfolio Algorithms with Hard Decoupling and Fail-Safe Governance.

---

## 1. Executive Summary & Defect Resolutions

Following the rigorous 3-Stage LLM Council Review, the original CASMI-Omega blueprint has been revised to systematically eliminate all 6 critical vulnerabilities identified during blind peer review:

| Vulnerability ID | Root Cause in v1 | Architectural Resolution in v2 |
| :--- | :--- | :--- |
| **C-1: VRAM Conflict** | MS-GPT 7B FP16 (~14GB) + FAISS-GPU (~2.8GB) exceeds 16GB VRAM | Use INT4 quantized generative model (~3.5–5GB) or sequential VRAM paging (CPU FAISS fallback). |
| **C-2: Track 3 Latency** | Unbounded beam search (30–80s/spectrum) blows the 9h budget 2–4× | Hard 10s per-spectrum timeout via `ThreadPoolExecutor`, `beam_width <= 3`, max 5 candidates. |
| **C-3: Runtime Governor Math** | `< 5.76s/molecule` constant was mathematically incorrect | Corrected to dynamic per-spectrum wall-clock tracking based on **21.3s / spectrum** (31,920s net / 1,500 spectra). |
| **C-4: Correlated Cliff Failure** | Tracks 2 & 3 both critically depended on MIST-CF top-1 formula | **Decoupled**: Softened Track 2 to top-3 formula posteriors + ±1H/±1O neighborhood; Track 3 gets formula-free fallback on high entropy; pre-compiled 1,000-SMILES COCONUT fallback. |
| **C-5: DreaMS Threshold Drift** | Flat 0.85 cosine cutoff ignored instrument family clustering | Calibrated per-instrument family threshold table (timsTOF-to-timsTOF prioritized); re-verify candidates in FP16. |
| **C-6: CSV Integrity Hazard** | Missing automated format/duplicate validation before write | Integrated `validate_submission()` ensuring exactly 25 unique InChIKey14s per row, no NaNs, correct delimiters. |

---

## 2. The Revised CASMI-Omega v2 Pipeline

```
                               Raw Multi-Energy LC-MS/MS Query
                                              │
                                              ▼
                [Module 1: Chemistry Physics & Preprocessing Engine]
                - 10-Adduct Monoisotopic Mass Deconvolution
                - Chemical Plausibility Filter: [M-2H2O+H]+ strictly requires Formula O >= 2
                - 13C Multi-Isotopologue Testing: evaluate M, M-1.003355, M-2.006710 Da
                - Canonical Neutral Mass Normalization for all queries
                                              │
                                              ▼
                [Module 1b: Multi-Energy Spectral Fusion Engine]
                - Process multi-collision energy ramps (e.g., 20, 35, 50 eV)
                - Per-energy DreaMS latent embedding extraction
                - Attention-weighted late pooling into unified query vector
                                              │
                                              ▼
                [Module 2: MIST-CF Soft Formula Prediction & Router]
                - Predicts top-3 formula posteriors (not a single brittle top-1 gate)
                - Expands search neighborhood: adds ±1H, ±1O formula variants
                - Calculates prediction entropy: routes to formula-free fallback if uncertain
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ▼                                     ▼                                     ▼
 [Track 1: Calibrated Library]        [Track 2: Soft DB Search]            [Track 3: Generative Hedge]
 - Tier 1: Query enveda-180           - Slice COCONUT candidates by        - MS-GPT / MS2Mol generative
   instrument-matched index             union of top-3 formula priors        sampling (INT4 quantized)
 - Tier 2: Query pan-library            and ±1H/±1O neighborhood           - Conditioned on top formula OR
   2.5M INT8 FAISS index              - Cross-modal / MIST 4096-bit          formula-free fallback if MIST
 - Instrument-stratified cosine         fingerprint Tanimoto scoring         posterior entropy is high
   verification in FP16               - Zero-hit fallback: pad from        - Strict 10s hard timeout;
 - High confidence (> thresh):          pre-compiled 1,000-SMILES            beam_width <= 3, max 5 cands
   retain for Rank 1 contention         COCONUT NP-diverse scaffolds         (graceful skip on timeout)
        │                                     │                                     │
        └─────────────────────────────────────┼─────────────────────────────────────┘
                                              ▼
                    [Module 3: Normalized Transductive Test-Set Networking]
                    - Operates strictly on Neutral-Mass-Normalized query spectra
                    - Unsupervised cosine similarity graph across all ~1,500 test spectra
                    - Propagates solved scaffolds across verified biosynthetic shifts:
                      • +162.0528 Da (Hexose)
                      • +132.0423 Da (Pentose / Arabinose / Xylose)
                      • +42.0106 Da (Acetyl)
                      • +14.0156 Da (Methyl / Homologation — class-conditional)
                    - Requires fragmentation co-validation (>= 2 shared shift fragments)
                                              │
                                              ▼
                    [Module 4: 30+ Feature GBDT LambdaMART Meta-Ranker]
                    Multi-signal learning across diverse representation spaces:
                    1. Spectral Similarity: DreaMS cosine, spectral entropy, dot product
                    2. Structural Fit: MIST Tanimoto, SpecBridge cross-modal score
                    3. Mass Defect & Chemistry: Precursor ppm error, absolute Da error, 13C score
                    4. Fragmentation Mechanics: Peak count match ratio, unexplained intensity fraction
                    5. Domain Priors: NP-likeness score (NPClassifier/RDKit), formula posterior rank
                    6. Orthogonality Indicators: Cross-track InChIKey14 agreement, track-of-origin
                    - Model validated via Cross-Library CV (holding out enveda-180)
                                              │
                                              ▼
                    [Module 5: Decision-Theoretic Slot Optimizer]
                    - Canonicalize all candidates to planar RDKit InChIKey14
                    - Strict deduplication (guarantees 25 unique structural skeletons)
                    - Greedy Expected-MRR Allocation using calibrated probabilities:
                      • Slots 1–3: High-confidence exploitation (Rank 1 = 1.0, Rank 2 = 0.5)
                      • Slots 4–12: Variant & structural hedge across orthogonal tracks
                      • Slots 13–25: Scaffolding diversification (COCONUT NP clusters)
                    - Fail-safe padding: Ensure full 25 slots from NP library if pool < 25
                                              │
                                              ▼
                    [Module 6: Submission Integrity & Runtime Governor]
                    - Pre-write validation: `validate_submission()` enforces 25 unique
                      InChIKey14s, semicolon delimiters, no NaNs/empties, all test IDs present
                    - Dynamic wall-clock runtime governor: targets 21.3s / spectrum with
                      8-minute reserve buffer for model loading and CSV verification
```

---

## 3. Concrete Architectural Fixes Detailed

### 3.1 Resolving Track Decoupling & The Single-Point-of-Failure (C-4)
- **The Problem:** In v1, if MIST-CF mispredicted the molecular formula, Track 2 found 0 molecules in COCONUT and Track 3 hallucinated structures with the wrong elemental composition.
- **The Solution:**
  1. **Top-3 Formula Union:** Track 2 queries COCONUT using an expanded filter consisting of the top 3 formulas predicted by MIST-CF, plus chemical neighbors ($\pm 1\text{H}, \pm 1\text{O}$). This eliminates the cliff drop-off.
  2. **Formula-Free De Novo Mode:** If the MIST-CF prediction has high entropy (Shannon entropy $H > 1.2$), Track 3 runs MS-GPT unconditioned on formula, directly sampling SMILES from the spectral embedding.
  3. **Universal Fallback Bank:** A pre-indexed database of 1,000 highly diverse natural product core scaffolds from COCONUT is bundled directly in the offline submission assets to guarantee zero empty slots.

### 3.2 Taming Latency and VRAM on Kaggle T4/L4 (C-1, C-2, C-3)
- **The Problem:** Unconstrained MS-GPT beam generation consumes 30–80s per spectrum and up to 14GB VRAM in FP16, colliding with FAISS-GPU and breaching the 9-hour limit.
- **The Solution:**
  1. **Quantization / Sizing:** Restrict MS-GPT to INT4 or INT8 weights (< 4.5 GB VRAM footprint), allowing it to comfortably coexist with the 2.8 GB INT8 FAISS index and DreaMS encoder within 16 GB VRAM.
  2. **Execution Timeout Guard:**
     ```python
     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
         future = executor.submit(generate_smiles, spectrum, beam_size=2, max_length=80)
         try:
             candidates = future.result(timeout=10.0)
         except concurrent.futures.TimeoutError:
             candidates = [] # Graceful fallback to Tracks 1 & 2
     ```
  3. **Correct Governor Budget:** Recalibrated budget is $21.3\text{ s/spectrum}$ (accounting for 480s cold start and verification margin), ensuring full utilization of the 9-hour Kaggle GPU session without timeout risks.

### 3.3 Neutral Mass Normalization & Biosynthetic Expansion (I-3, I-4)
- **The Problem:** Calculating mass differences between raw $m/z$ values conflates adduct ionization shifts ($+21.982\text{ Da}$ for $\text{Na}^+ - \text{H}^+$) with biological transformations, creating spurious network edges.
- **The Solution:** All queries are deconvoluted to neutral monoisotopic mass before building the transductive test-set cosine network.
- **Enriched Delta Library:** Added $+132.0423\text{ Da}$ (Pentose: Arabinose/Xylose, critical for plant flavonoid glycosides) alongside Hexose ($+162.0528\text{ Da}$), Acetyl ($+42.0106\text{ Da}$), and Methylation ($+14.0156\text{ Da}$).

### 3.4 Submission Integrity Protection (C-6)
- **Pre-Submission Invariant Check:** The submission code guarantees zero disqualifications or corrupted scores by enforcing:
  ```python
  def validate_submission(df, expected_ids):
      assert len(df) == len(expected_ids), "Missing spectrum IDs in submission"
      assert set(df['id']) == set(expected_ids), "Mismatch in spectrum IDs"
      for idx, row in df.iterrows():
          cands = row['candidates'].split(';')
          assert len(cands) == 25, f"Row {idx} has {len(cands)} candidates, expected exactly 25"
          assert len(set(cands)) == 25, f"Row {idx} contains duplicate InChIKey14 entries"
          assert all(len(c) == 14 and c.isalnum() for c in cands), f"Row {idx} has invalid InChIKey14"
      return True
  ```

---

## 4. Summary

CASMI-Omega v2 addresses every criticism raised by the LLM Council while preserving the core advantages that won unanimous support:
1. **Three complementary retrieval layers** (Library FAISS, Soft DB search, Generative hedge).
2. **Unorthodox transductive test networking** across plant metabolomics congener series.
3. **Rigorous decision-theoretic slot allocation** optimizing specifically for Kaggle's MRR@25 on InChIKey14.
4. **Hard runtime and memory boundaries** ensuring safe execution within Kaggle's 9h / 16GB VRAM sandbox.
