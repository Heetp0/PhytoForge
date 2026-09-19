# Enveda CASMI 2026: Molecule ID From Mass Spectra

Kaggle Competition: [Enveda - CASMI 2026 Molecule Identification from Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)

## Objective
Predict 2D chemical structures (SMILES) of small molecules from LC-MS/MS mass spectra, evaluated by Mean Reciprocal Rank @ 25 (MRR@25) on RDKit tautomer-canonicalized InChIKey14.

## Evaluation Metric
$$
\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{\text{rank}_u}
$$
- Ground truth match: `InChIKey14` (first block of InChIKey, matching 2D skeleton / connectivity, ignoring tautomers and stereocenters).
- Top 25 guesses per molecule, semicolon-separated, best guess first.

## Directory Structure
```
enveda-casmi-2026/
├── configs/          # Experiment & hyperparameter configuration files
├── data/             # Raw & processed data (train.parquet, test.parquet, databases)
├── models/           # Checkpoints, embeddings, and pre-trained weights
├── notebooks/        # Jupyter/Kaggle exploratory and submission notebooks
├── src/              # Python source packages
│   ├── data/         # Data loaders, parquet readers, spectral cleaners
│   ├── chemistry/    # Adduct calculator, InChIKey14 matcher, formula generator
│   ├── retrieval/    # Spectral library search & vector index (Cosine, DreaMS)
│   ├── reranking/    # Candidate scoring & in-silico MS/MS matchers
│   └── submission/   # Submission validator and CSV formatter
└── submissions/      # Generated submission.csv files
```
