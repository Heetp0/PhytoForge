# Unorthodox & High-Leverage Competitive Methodologies
**Project:** Enveda CASMI 2026 (Kaggle) — Molecule ID from Mass Spectra  
**Objective:** Exploit unique domain physics, graph recombination, metric portfolio theory, and transductive test networking to beat conventional pipelines.

---

## 1. Unorthodox Strategy 1: The "Lego Brick" Knapsack Graph Assembler (BRICS + ILP)

### The Problem with Pure LLMs (SMILES GPTs)
Autoregressive token generators (SMILES transformers) frequently suffer from:
- Syntax hallucination (unclosed rings, invalid valence).
- Missing bridgehead rules (Bredt's rule violations).
- Disconnect from observed fragment masses.

### The Unorthodox Alternative
Instead of generating text characters, assemble molecules from **verified natural-product substructure blocks (Lego bricks)**:

```
[Observed MS2 Peaks] ---> [Match Against Precomputed BRICS Library]
(e.g., m/z 163.04, 179.03, 301.03)      (Phenolic rings, sugars, terpene units)
                                                       │
                                                       ▼
                               [Integer Linear Programming (ILP)]
                                 Sum(c_i * mass_i) = M_precursor
                                                       │
                                                       ▼
                              [BRICS Connection Rule Stitching]
                              (Valid chemical bond permutations)
```

1. **Brick Extraction:** Decompose 275,000 training molecules into substructure fragments using RDKit's **BRICS** (Breaking of Retrosynthetically Interesting Chemical Substructures) and **RECAP**.
2. **Mass-Knapsack Solver:** Formulate candidate formula composition as an exact Integer Linear Program (ILP):
   $$\text{Minimize } \left| \sum_{j=1}^{M} c_j \cdot m_j - M_{\text{precursor}} \right| \quad \text{subject to } c_j \in \{0, 1, 2, \dots\}, \, \text{error} < 5\text{ ppm}$$
3. **Graph Stitching:** Combine selected bricks along compatible BRICS link types.
4. **Why it wins:** Every candidate generated is **100% chemically valid**, retrosynthetically plausible, and guaranteed to contain the exact substructures that produced the observed MS/MS peaks.

---

## 2. Unorthodox Strategy 2: Transductive Molecular Networking on the Test Set

### The Observation
In natural product extracts (which Enveda specializes in), molecules rarely appear in isolation. They occur in **metabolic families / biosimilar series** (e.g., aglycones alongside mono-glucoside and di-glucoside derivatives; methylated/hydroxylated congeners).

### The Unorthodox Hack
The test set contains ~1,500 spectra across ~400 molecules. Run **unsupervised spectral networking (GNPS algorithm)** across the *unlabeled test set*:

$$
\Delta m_{AB} = m_{\text{prec}}(A) - m_{\text{prec}}(B)
$$

- If Molecule $A$ and Molecule $B$ share identical high-energy fragments, but $m/z(B) - m/z(A) = 162.0528\text{ Da}$ (exact neutral loss of a hexose sugar) or $14.0156\text{ Da}$ ($\text{CH}_2$ methylation):
  - If Molecule $A$ is identified with high confidence in Class 1 (Library) or Class 2 (COCONUT), **Molecule $B$ can be directly synthesized by attaching the functional group to Molecule $A$**.
- This turns an impossible Class 3 novel molecule into a simple analog derivation of a solved Class 1 molecule.

---

## 3. Unorthodox Strategy 3: MRR@25 Maximum Marginal Relevance (MMR) Portfolio Optimization

### The Problem
Most competitors rank candidates greedily by individual probability:
$$\text{Rank } 1..25 = \text{Top-25 } P(c_i \mid \text{spectrum})$$
If the top 10 candidates are all subtle positional isomers of the same quercetin scaffold, they share 95% chemical similarity. If that scaffold is wrong, **all 10 slots are wasted ($0\text{ points}$)**.

### The Unorthodox Metric Hack
Treat the 25 submission slots like an **investment portfolio** using a rank-decayed **Maximum Marginal Relevance (MMR)** criterion:

$$\text{Slot } k = \arg\max_{c \notin \mathcal{S}_{k-1}} \left[ \lambda_k \cdot \text{Score}(c) - (1 - \lambda_k) \max_{s \in \mathcal{S}_{k-1}} \text{Tanimoto}(c, s) \right]$$

- **Rank 1–3 ($\lambda \approx 1.0$):** Exploit mode. Place highest-probability candidates.
- **Rank 4–12 ($\lambda \approx 0.7$):** Conservative diversification. Plausible structural analogs.
- **Rank 13–25 ($\lambda \approx 0.4$):** Aggressive exploration. Orthogonal molecular scaffolds that still satisfy the precursor mass and formula.
- **Result:** Mathematically maximizes the probability that **at least one** guess in the top 25 captures the correct `InChIKey14`.

---

## 4. Unorthodox Strategy 4: Ion Mobility & Collision Cross Section (CCS) Constraints

### The Advantage of Bruker timsTOF
Enveda collected the test set on a **Bruker timsTOF**. Unlike standard mass specs, timsTOF measures **ion mobility ($1/K_0$)**, which directly translates to **Collision Cross Section (CCS in $\text{Å}^2$)**:
- Flat, planar aromatic molecules have small CCS.
- Branched, bulky, 3D globular molecules have large CCS.

### The Unorthodox Application
If the competition dataset contains ion mobility or peak drift time:
- Train a lightweight GNN (or use pre-trained **CCSP** / **DeepCCS**) to predict theoretical CCS for candidate structures in milliseconds.
- Filter or re-rank candidates: any candidate whose 3D conformational size does not match the observed mobility peak is downweighted or pruned.

---

## 5. Unorthodox Strategy 5: Test-Time Instrument Alignment (TTT)

### The Domain Shift Problem
Training spectra come from Orbitrap, Synapt, and public instruments, while the test set is exclusively Bruker timsTOF.
- Even identical molecules have shifted peak intensity ratios due to quadrupole transmission differences.

### The Unorthodox Fix
Before running final test inference, run a self-supervised **Test-Time Training (TTT)** pass on the unlabeled test spectra:
- Use masked peak prediction on the test spectra with a low learning rate ($10^{-5}$) for 5 epochs.
- The encoder adapts its batch-norm statistics and attention weights to the exact instrument calibration, noise floor, and detector gain of the hidden test instrument.
