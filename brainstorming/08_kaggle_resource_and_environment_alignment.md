# Kaggle Platform Resources & Environment Alignment Spec
**Project:** Enveda CASMI 2026 (Kaggle) — Molecule ID from Mass Spectra  
**Document Purpose:** Define exact Kaggle hardware, operating limits, storage budgets, offline execution constraints, and environment packaging rules to guide the implementation plan.

---

## 1. Hardware & Execution Constraints

| Resource | Kaggle Specification | CASMI-Omega Allocation / Budget | Margin / Safety Factor |
| :--- | :--- | :--- | :--- |
| **GPU Accelerator** | 1× NVIDIA Tesla T4 (16 GB GDDR6) or 1× NVIDIA L4 (24 GB) | Peaks at **~9.4–10.5 GB VRAM** (INT4 MS-GPT + INT8 FAISS + DreaMS + CUDA activations) | **> 5.5 GB VRAM headroom** |
| **Host CPU** | 2 to 4 vCPUs (Intel Xeon / AMD EPYC) | 4 worker threads for candidate processing, I/O streaming | Negligible CPU load |
| **System RAM** | **~29–31 GB** host RAM | ~6 GB for COCONUT sqlite/dict index + ~3 GB FAISS CPU mirror + ~2 GB metadata | **> 18 GB RAM headroom** |
| **Session Disk (`/kaggle/working`)**| **20 GB** writeable scratch space | ~50 MB (`submission.csv` is ~1.5 MB; scratch logs ~10 MB) | **> 19.9 GB disk headroom** |
| **Mounted Input (`/kaggle/input`)** | Up to **100 GB** total attached datasets (max 20 GB per single dataset) | **~12–16 GB total** across 2 custom offline datasets (models, indices, wheels) | Comfortably fits within limits |
| **Max Wall-Clock Time** | **9.0 Hours (32,400 seconds)** hard kill | Bounded to **~3.5–6.0 Hours** total execution for 1,500 spectra | **> 3.0 Hours time headroom** |
| **Internet Access** | **DISABLED** on submission runs (Code Competition rule) | 100% offline self-contained (all pip wheels, weights, and DBs pre-bundled) | Compliant |

---

## 2. Submission & Evaluation Mechanics

### 2.1 The Two-Phase Hidden Test Set
- **Public Run (Interactive / Commit):** The notebook runs on the visible test set (`test.parquet`, ~1,500 spectra, ~400 molecules).
- **Private Evaluation (Kaggle Scoring Worker):** Upon pressing "Submit", Kaggle swaps `test.parquet` with the **hidden private test set** (expected ~1,500–4,000 spectra) and reruns the entire notebook from scratch in an isolated sandbox with **internet disabled**.
- **Crucial Rule:** The notebook must dynamically detect the length of `test.parquet` and adjust the per-spectrum wall-clock budget dynamically.

### 2.2 Submission CSV Contract
- **File Path:** `/kaggle/working/submission.csv`
- **Columns:** `id`, `candidates`
- **Candidate Delimiter:** Semicolon (`;`)
- **Row Invariant:** Exactly 25 unique, valid 14-character planar InChIKey strings (`InChIKey14`).
- **Target Metric:** **MRR@25 on InChIKey14**
  $$MRR@25 = \frac{1}{N}\sum_{i=1}^N \frac{1}{\text{rank}_i} \quad (\text{rank} \le 25, \text{else } 0)$$

---

## 3. Offline Packaging Strategy (Kaggle Datasets)

Because internet is strictly disabled during submission, all dependencies, pretrained neural network checkpoints, pre-indexed vector libraries, and candidate databases must be uploaded as private Kaggle Datasets and attached to the notebook.

### Dataset 1: `casmi-omega-wheels` (~500 MB)
Pre-downloaded Python `.whl` files matching Kaggle's Python 3.10/3.11 Linux x86_64 environment:
- `rdkit` (cheminformatics engine)
- `faiss-gpu` or `faiss-cpu` (vector similarity search)
- `matchms` (mass spectrometry utility)
- `bitsandbytes` / `autoawq` (INT4 model quantization runtime)
- `lightgbm` / `xgboost` (GBDT meta-ranker)
- Installation in notebook:
  ```bash
  !pip install --no-index --find-links=/kaggle/input/casmi-omega-wheels/ rdkit faiss-gpu matchms bitsandbytes
  ```

### Dataset 2: `casmi-omega-weights` (~8–12 GB)
Pre-trained model weights and quantized checkpoints:
- **DreaMS Checkpoint:** `dreams_pretrained.pt` (~0.8 GB, from Zenodo 10997887)
- **MIST-CF Checkpoint:** `mist_cf_weights.pt` (~0.1 GB)
- **MS-GPT INT4 Model:** `ms_gpt_int4/` (~3.8 GB)
- **GBDT LambdaMART Model:** `lambda_mart_reranker.lgb` (~15 MB)

### Dataset 3: `casmi-omega-indices` (~4–6 GB)
Pre-computed search databases and vector indices:
- **FAISS INT8 Index (2.5M spectra):** `train_spectra_int8.faiss` (~2.8 GB)
- **enveda-180 Sub-Index:** `enveda_180_int8.faiss` (~150 MB)
- **COCONUT Formula-Indexed SQLite / Feather DB:** `coconut_candidates.feather` (~1.2 GB, indexed by neutral formula)
- **Pre-compiled NP Diversity Fallback:** `coconut_1000_diverse_scaffolds.parquet` (~5 MB)

---

## 4. Cold-Start vs Hot-Inference Budget Alignment

```
9.0 Hours Total Budget (32,400s)
│
├── [Phase 0: Cold Start & Asset Mounting] (~300–480s / 5–8 min)
│   ├── Install offline wheels: ~30s
│   ├── Load DreaMS & MIST-CF models into GPU: ~15s
│   ├── Load INT4 MS-GPT into GPU: ~45s
│   ├── Memory-map FAISS index & COCONUT database: ~60s
│   └── Warm-up dummy forward pass (prevent PyTorch JIT stall): ~10s
│
├── [Phase 1: Batch Preprocessing & Embedding Extraction] (~60–90s)
│   ├── Neutral mass deconvolution & 10-adduct conversion for 1,500 spectra: ~2s
│   ├── Multi-Energy DreaMS embedding extraction (batch size 64): ~15s
│   ├── MIST-CF soft formula top-3 posterior prediction: ~30s
│   └── Compute 1,500 x 1,500 Transductive Adjacency Matrix: ~1s
│
├── [Phase 2: Per-Query Retrieval, Generation & Re-Ranking] (~12,000–18,000s / 3.3–5.0h)
│   ├── Dynamic per-spectrum budget: ~21.3s max
│   ├── Track 1 (FAISS enveda-180 + Pan-Library): ~0.05s
│   ├── Track 2 (Soft COCONUT formula search + Tanimoto): ~1.2s
│   ├── Track 3 (MS-GPT INT4, beam=2, max_len=80, timeout=10s): ~2.5–5.0s
│   ├── Module 3 (Transductive scaffold propagation): ~0.1s
│   ├── Module 4 (30+ Feature GBDT LambdaMART scoring): ~0.05s
│   └── Module 5 (Planar InChIKey14 deduplication & slot allocation): ~0.05s
│       Total average nominal latency: ~4.0–6.5s per spectrum
│
├── [Phase 3: Validation & Submission Integrity Check] (~30s)
│   ├── Run validate_submission(df, test_ids): ~5s
│   └── Write /kaggle/working/submission.csv: ~10s
│
└── [Reserve Buffer]: ~10,000s (~2.8 Hours unallocated safety margin)
```

---

## 5. Local Hardware vs Kaggle Alignment

| Environment | Local Development Rig | Kaggle Target Environment | Adaptation Strategy |
| :--- | :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4060 (8 GB VRAM) | NVIDIA Tesla T4 (16 GB) / L4 (24 GB) | Develop with CPU FAISS & INT4 MS-GPT locally; fits inside 8 GB VRAM. Runs even faster with 16 GB on Kaggle. |
| **CPU / RAM** | Local 16–32 GB RAM | 30 GB RAM | Memory-map databases (`mmap=True`) so OS caches pages without hitting RAM limits. |
| **OS / Shell** | Windows 11 / PowerShell | Ubuntu 22.04 / Bash | Code written in pure, OS-agnostic Python (`pathlib.Path`, forward slashes, cross-platform PyTorch). |
