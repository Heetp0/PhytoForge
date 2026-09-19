# PhytoForge: Botanical tandem mass spectrometry annotation

PhytoForge is a molecular structure identification pipeline developed for the [Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra). The task requires predicting chemical structures from liquid chromatography tandem mass spectrometry (LC-MS/MS) data acquired on Bruker timsTOF instruments.

## Evaluation metric

Submissions are scored using Mean Reciprocal Rank at 25 (MRR@25) on RDKit tautomer-canonicalized InChIKey14:

$$
\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{\text{rank}_u}
$$

- Ground truth matching uses `InChIKey14`, the first 14-character block representing two-dimensional planar connectivity without stereochemistry.
- Predictions contain up to 25 semicolon-separated candidate keys per test spectrum.

## System architecture

The pipeline consists of five interconnected modules designed to run within Kaggle offline constraints (16 GB GPU memory and 9 hours total runtime):

1. Preprocessing and deconvolution
   - Checks precursor adduct feasibility, requiring at least two oxygen atoms for neutral water loss.
   - Deconvolutes carbon-13 multi-isotopologue patterns on high-mass ions.
2. Tri-track candidate generation
   - Track 1: Instrument-stratified cosine search over 1024-dimensional DreaMS embeddings derived from attention-weighted multi-energy collision spectra (20, 35, and 50 eV).
   - Track 2: Formula-conditioned database querying using MIST-CF top-3 chemical formula posteriors and vectorized 4096-bit Morgan fingerprint similarities.
   - Track 3: Bounded generative de novo sampling executed under a strict 10-second timeout.
3. Transductive botanical networking
   - Connects unannotated spectra across the test set using characteristic plant secondary metabolism mass shifts (+132.0423 Da for pentosyl, +162.0528 Da for hexosyl, and +146.0579 Da for rhamnosyl modifications).
   - Validates candidate links against shared MS/MS fragment peaks.
4. Meta-ranking and portfolio optimization
   - Extracts 32 features spanning spectral similarity, formula probability, and molecular physical descriptors.
   - Scores candidates using a LambdaMART gradient boosted decision tree.
   - Allocates 25 output slots through expected reciprocal rank optimization to balance top-rank precision and structural diversity.
5. Runtime governance and submission validation
   - Manages execution time dynamically with a 21.3 seconds per spectrum budget governor.
   - Enforces formatting rules: exactly 25 unique alphanumeric InChIKey14 values per query, semicolon delimiters, and zero null values.

## Installation and usage

### Environment setup
```bash
# Clone the repository
git clone https://github.com/enveda/phytoforge.git
cd phytoforge

# Install dependencies
pip install -r requirements.txt
```

### Running the pipeline
```bash
# Execute end-to-end inference on test spectra
python -m src.submission.pipeline \
  --input data/test.parquet \
  --output submissions/submission.csv \
  --governor 21.3 \
  --vram-cap 16
```

### Validating submissions
```bash
# Verify format integrity before submission
python -m src.submission.validator \
  --submission submissions/submission.csv \
  --expected-ids data/test_ids.txt
```

## Repository structure

```
enveda-casmi-2026/
├── configs/          # Configuration files and hyperparameter definitions
├── data/             # Input parquet files, spectral databases, and reference indices
├── docs/             # Technical specifications, gates ledger, and brand guidelines
│   ├── brand-guidelines.md
│   └── superpowers/
│       ├── plans/    # Implementation plans
│       └── specs/    # Architecture and design specifications
├── models/           # DreaMS, MIST-CF, and LambdaMART model weights
├── notebooks/        # Data exploration and local validation notebooks
├── src/              # Core pipeline packages
│   ├── chemistry/    # Adduct calculator, formula utils, and InChIKey tools
│   ├── data/         # Preprocessors, parquet readers, and transductive network
│   ├── reranking/    # LambdaMART ranker, feature extractor, and slot optimizer
│   ├── retrieval/    # DreaMS search, multi-energy fusion, and de novo engine
│   └── submission/   # Runtime governor, pipeline runner, and submission validator
├── submissions/      # Generated competition submission files
└── tests/            # Test suite covering all 12 pipeline acceptance gates
```

## Verification

The codebase includes an automated test suite verifying all 12 engineering acceptance gates:

```bash
# Run the complete test suite
pytest tests/ -v
```

Test results:
- 16 test modules
- 256 passed assertions
- 0 failures, 0 warnings
