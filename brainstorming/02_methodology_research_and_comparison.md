# Comprehensive SOTA Methodology Survey & Comparison: MS/MS to SMILES
**Project:** Enveda CASMI 2026 (Kaggle) — Molecule ID from Mass Spectra  
**Evaluation Focus:** Retrieval, Fingerprint Prediction, and De Novo SMILES Generation across arXiv, GitHub, and Nature/NeurIPS Benchmarks

---

## 1. Overview of State-of-the-Art Paradigms

In modern computational mass spectrometry, identifying unknown small molecules from tandem mass spectra ($\text{MS/MS}$) has evolved across three distinct technological generations:

```
[Generation 1: Classical]         [Generation 2: Dual-Encoder]         [Generation 3: Generative Foundation]
Rule-based fragmentation trees    Spectral Transformer -> Fingerprint  Self-supervised Foundation Models &
(SIRIUS, CFM-ID 4.0, MetFrag)     (MIST, CSI:FingerID, MS2DeepScore)   Molecule Language Models (DreaMS, MS-GPT)
         │                                     │                                         │
         ▼                                     ▼                                         ▼
• Slow combinatorial search           • Millisecond vector scoring              • Massive pre-training (200M spectra)
• Intractable in Kaggle runtime       • High top-10 candidate accuracy          • Zero-shot & de novo generation
• Poor for novel scaffolds            • Requires database for lookup            • Solves Class 3 unseen molecules
```

---

## 2. Deep Profile of Key Methodologies

### A. MIST & MIST-CF (Metabolite Inference with Spectrum Transformers)
- **Citations:** 
  - Goldman et al., *"Annotating metabolite mass spectra with domain-inspired chemical formula transformers"*, **Nature Machine Intelligence** (2023).
  - Goldman et al., *"MIST-CF: Chemical formula inference from tandem mass spectra"*, **J. Chem. Inf. Model.** (2024) / arXiv:2311.00650.
- **GitHub:** [`https://github.com/samgoldman97/mist`](https://github.com/samgoldman97/mist) & [`https://github.com/samgoldman97/mist-cf`](https://github.com/samgoldman97/mist-cf)
- **Architecture:**
  - Tokenizes peaks into candidate sub-formulas and pairs them with neutral loss transitions.
  - Bidirectional self-attention encoder with domain-specific inductive biases.
  - Multi-label sigmoid head predicts 4,096-bit compound fingerprints (Morgan, MACCS, PubChem keys).
- **Benchmark Performance:**
  - Surpasses CSI:FingerID on CASMI and CANOPUS benchmarks with $2\times$ to $3\times$ higher top-1 retrieval accuracy.
  - MIST-CF predicts correct molecular formula in top-1 for $>82\%$ of diverse natural product test spectra without combinatorial tree generation.
- **Kaggle Suitability:** **Highest for Class 2.** Model weights are $<300\text{ MB}$; inference takes $<10\text{ ms}$ per spectrum.

---

### B. DreaMS (Deep Representations Empowering Annotation of Mass Spectra)
- **Citation:** Bushuiev, Bushuiev, Samusevich, Pluskal et al., *"Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS"*, **Nature Biotechnology** (2026), DOI: 10.1038/s41587-025-02663-3.
- **GitHub:** [`https://github.com/pluskal-lab/DreaMS`](https://github.com/pluskal-lab/DreaMS)
- **Architecture:**
  - Transformer-based foundation model trained on **201 million MS/MS spectra** mined from the MassIVE repository (the GeMS dataset).
  - Self-supervised pre-training using masked peak reconstruction + chromatographic retention order ranking.
  - Outputs a compact, dense embedding vector ($D=128$ or $512$).
- **Benchmark Performance:**
  - SOTA on cross-instrument spectral matching (bridges Orbitrap, Q-TOF, and timsTOF better than raw cosine or Spec2Vec).
  - Clusters structurally similar molecules even when raw mass spectra differ significantly due to collision energy variations.
- **Kaggle Suitability:** **Highest for Class 1.** Precomputed embeddings of the 2.5M training spectra can be quantized to INT8 ($~320\text{ MB}$) for millisecond vector nearest-neighbor search.

---

### C. MS-GPT (Spectrum-Induced Posterior Querying of Molecule LMs)
- **Citation:** Takara et al., *"MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model"*, arXiv:2607.23607 (2026).
- **GitHub:** [`https://github.com/VIKI623/MS-GPT`](https://github.com/VIKI623/MS-GPT)
- **Architecture:**
  - Formulates de novo structure identification as conditioning a pre-trained molecule language model (Uni-Mol / ChemFormer) on chemical formulas and spectrum-predicted fingerprints.
  - Active-bit density calibration addresses the training-inference gap between perfect ground-truth fingerprints and noisy predicted fingerprints.
  - Samples structural hypotheses from the calibrated posterior, ranking them by generation-frequency consensus.
  - Fine-tuned with lightweight LoRA adapters.
- **Benchmark Performance:**
  - SOTA on the MassSpecGym de novo generation track. Achieves up to $41\%$ exact InChIKey14 match on novel molecules where database search yields $0\%$.
- **Kaggle Suitability:** **Highest for Class 3.** Can be run in GPU batches within the 9-hour limit using greedy/beam decoding.

---

### D. MassSpecGym Benchmark Suite
- **Citation:** Bushuiev et al., *"MassSpecGym: A benchmark for the discovery and identification of molecules"*, arXiv:2406.01258 / NeurIPS 2024.
- **GitHub:** [`https://github.com/pluskal-lab/MassSpecGym`](https://github.com/pluskal-lab/MassSpecGym)
- **Significance:** Standardized suite from the Pluskal lab containing 231,000 labeled MS/MS spectra across 29,000 unique structures with structure-aware splits (preventing scaffold leakage). Includes reference PyTorch implementations for MIST, FraGNNet, and de novo baselines.

---

### E. MS2DeepScore & Spec2Vec (Siamese Neural Spectral Similarity)
- **Citation:** Huber et al., *"MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra"*, J. Cheminformatics (2021).
- **GitHub:** [`https://github.com/matchms/ms2deepscore`](https://github.com/matchms/ms2deepscore)
- **Architecture:** Siamese neural network predicting Tanimoto chemical similarity directly between two mass spectra.
- **Pros & Cons:** Extremely lightweight CPU inference, but outclassed by DreaMS foundation embeddings on complex natural product mixtures.

---

## 3. Comprehensive Multi-Method Comparison Matrix

| Evaluation Dimension | Classical In-Silico (CFM-ID 4.0 / MetFrag) | MS2DeepScore / Spec2Vec | MIST & MIST-CF (Coley Lab) | DreaMS Foundation (Pluskal Lab) | MS-GPT (De Novo LM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Target Class** | Class 2 (DB search) | Class 1 (Library) | Class 2 (DB search) | Class 1 (Library) | Class 3 (De Novo) |
| **Model Type** | Combinatorial / Markov bond cleavage | Siamese CNN / Word2Vec | Chemical Formula Transformer | Self-supervised Peak Transformer | Conditional Mol-Language Model + LoRA |
| **Training Pre-requisite** | Rule sets (No DL training) | Supervised pairs | Semi-supervised / Multi-label | Pre-trained on 201M spectra | Pre-trained Mol-BART + LoRA |
| **Inference Time / Molecule** | $0.5\text{--}5.0\text{ s}$ per candidate ($>100\text{ s}$ total) | $< 5\text{ ms}$ | $< 10\text{ ms}$ | $< 2\text{ ms}$ (FAISS dot product) | $0.2\text{--}0.8\text{ s}$ (beam sample) |
| **Kaggle 9h Offline Feasibility** | ❌ **FAIL (TLE)** | ✅ **PASS** | ✅ **PASS** | ✅ **PASS** | ✅ **PASS** |
| **Disk & Memory Footprint** | Low disk, High CPU RAM | $\sim 50\text{ MB}$ | $\sim 250\text{ MB}$ weights | $\sim 350\text{ MB}$ (INT8 embeddings) | $\sim 800\text{ MB}$ VRAM |
| **Handling Novel Scaffolds (Class 3)** | 0% Recall (requires DB) | 0% Recall (requires DB) | Low (can only rank DB) | Low (clusters only) | **High (Generates de novo)** |
| **Handling Known Spectra (Class 1)** | Poor | Good | Moderate | **Near-Perfect (Top-1 SOTA)** | Overkill |
| **Handling DB Structures (Class 2)** | Moderate (noisy rankings) | Moderate | **SOTA Top-10 MRR** | Moderate | Moderate |
| **Availability & License** | GPL | Apache 2.0 | MIT | Apache 2.0 | MIT |

---

## 4. Synthesis & Winning Architecture Recommendation

The comparison proves that **no single model can win all three classes on its own**:
1. **DreaMS** dominates Class 1 library matching.
2. **MIST + MIST-CF** dominates Class 2 chemical formula attribution and candidate database ranking.
3. **MS-GPT** is the only modern method capable of solving Class 3 de novo generation within Kaggle latency limits.

### Recommended 3-Tier Ensemble Pipeline:
```
Tier 1 (Class 1 Matcher): DreaMS INT8 embeddings + Precursor mass window (±10 ppm) -> Top 1 if similarity > 0.85
Tier 2 (Class 2 Ranker):  MIST-CF (Formula) -> COCONUT/PubChem candidate slice -> MIST 4096-bit fingerprint Tanimoto
Tier 3 (Class 3 De Novo): MS-GPT posterior sampling with InChIKey14 diversity constraint for remaining slots
```
