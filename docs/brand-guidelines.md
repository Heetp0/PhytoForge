# Brand guidelines: PhytoForge

## Project overview
PhytoForge is an open-source software library for identifying molecular structures from tandem mass spectrometry (MS/MS) data. It is developed for the Enveda CASMI 2026 competition, which evaluates predictions by Mean Reciprocal Rank at 25 (MRR@25) on RDKit tautomer-canonicalized planar InChIKey14 strings.

## Name and naming rules

### Primary names
- Software package and CLI: `phytoforge`
- Repository title: `phytoforge` or `enveda-phytoforge`
- Competition solution and paper title: `PhytoForge-14`

### Name origin
- `Phyto`: Refers to plant secondary metabolites and the botanical mass-shift networks used to link derivatives to known aglycone cores.
- `Forge`: Refers to constructing candidate molecules through targeted de novo generation and database querying.
- `-14`: Identifies the target evaluation metric, the 14-character planar InChIKey block that captures two-dimensional skeletal connectivity.

### Rules for using the name
- Use lowercase `phytoforge` when referring to the Python package, CLI tool, or directory path.
- Use camel case `PhytoForge` in written documentation, presentation slides, and formal writeups.
- Append `-14` as `PhytoForge-14` when discussing the competition submission, technical report, or benchmark results.
- Do not add hyphenated competition tags like `phytoforge-casmi2026` to package names.

## Positioning statement
PhytoForge identifies chemical structures from high-resolution Bruker timsTOF MS/MS data within offline resource budgets (16 GB VRAM and a 9-hour total execution limit). It processes stepped collision energy frames (20, 35, and 50 eV) and performs precursor adduct deconvolution, requiring at least one oxygen atom for single water loss and two for double water loss. The system combines late-pooled 1024-dimensional DreaMS embeddings, MIST-CF chemical formula routing with neighborhood expansion, transductive botanical mass-shift propagation (+132.0423 Da pentosyl, +162.0528 Da hexosyl, and +146.0579 Da rhamnosyl modifications), and bounded autoregressive generative sampling under a 10-second timeout. Candidate structures are scored with a 32-feature LambdaMART meta-ranker and assembled using a decision-theoretic expected MRR@25 portfolio optimizer to output exactly 25 unique InChIKey14 candidates per spectrum.

## Taglines
- Primary: Transductive botanical MS/MS deconvolution and portfolio ranking for plant metabolomics.
- Short: InChIKey14 structural identification from tandem mass spectra.

## Tone of voice
- Direct and factual: State mechanics, limits, and measured numbers plainly.
- Precise chemistry and mass spectrometry terminology: Use standard terms such as precursor ion, collision-induced dissociation, adduct, isotopologue, and planar InChIKey.
- Zero promotional phrasing: Avoid words like groundbreaking, cutting-edge, revolutionary, seamless, or robust. State what the algorithm does and report test results directly.

## Repository and CLI conventions

### Package import
```python
import src as pf
from src.pipeline import PhytoForgePipeline
```

### Command-line interface
```bash
# Run the pipeline under runtime and memory limits
python -m src.submission.pipeline --input data/test.parquet --output submissions/submission.csv --governor 21.3 --vram-cap 16

# Validate submission formatting against competition rules
python -m src.submission.validator --submission submissions/submission.csv --expected-ids data/test_ids.txt
```

### Color palette
The visual identity uses neutral lab tones paired with botanical accents:
- Primary Forest: `#1E4D2B` (Headers, brand mark)
- Charcoal Slate: `#1F2421` (Body text, terminal background)
- Ion Blue: `#2E6F9E` (Spectral peaks, links, active states)
- Pure White: `#FFFFFF` (Surface background)
- Border Gray: `#DCE1DE` (Card outlines, table borders)
