"""
Synthetic Spectrum and Chemical Dataset Generator for CASMI 2026.
Generates realistic LC-MS/MS spectra, precursor m/z, isotopic 13C mispicks,
and mock library / candidate databases for benchmarking.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.chemistry.adducts import (
    CARBON_13_DELTA,
    COMPETITION_ADDUCTS,
    calculate_precursor_mz as chem_calc_precursor_mz,
)
from src.data.loader import QuerySpectrum, SpectrumData

# Curated Canonical Archetype Molecules with authentic MS2 fragment ions
ARCHETYPE_COMPOUNDS: List[Dict[str, Any]] = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "formula": "C9H8O4",
        "neutral_mass": 180.0422587,
        "inchikey14": "BSYNRYMUTXBXSQ",
        "class": "NSAID",
        "fragments": [138.0317, 120.0211, 92.0262, 77.0391],
    },
    {
        "name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "formula": "C8H10N4O2",
        "neutral_mass": 194.0803756,
        "inchikey14": "RYYVLZVUVIJVGH",
        "class": "Xanthine",
        "fragments": [137.0581, 109.0632, 82.0524, 67.0291],
    },
    {
        "name": "Paracetamol",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "formula": "C8H9NO2",
        "neutral_mass": 151.0633285,
        "inchikey14": "RZVAJINKPMORJF",
        "class": "Analgesic",
        "fragments": [110.0475, 109.0396, 81.0447, 65.0386],
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "formula": "C13H18O2",
        "neutral_mass": 206.1306798,
        "inchikey14": "HEFNNWSXXWATRW",
        "class": "NSAID",
        "fragments": [161.1325, 119.0855, 91.0542, 105.0699],
    },
    {
        "name": "Quercetin",
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "formula": "C15H10O7",
        "neutral_mass": 302.0426527,
        "inchikey14": "REFJWTPEDVJJIY",
        "class": "Flavonol",
        "fragments": [153.0182, 137.0233, 179.0339, 229.0495],
    },
    {
        "name": "Rutin",
        "smiles": "CC1OC(OCC2OC(Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)C(O)C(O)C2O)C(O)C(O)C1O",
        "formula": "C27H30O16",
        "neutral_mass": 610.1533816,
        "inchikey14": "OGNKVOVYSYLLOY",
        "class": "Flavonoid Glycoside",
        "fragments": [303.0499, 153.0182, 465.1027, 302.0427],
    },
    {
        "name": "Nicotinic Acid",
        "smiles": "c1ccncc1C(=O)O",
        "formula": "C6H5NO2",
        "neutral_mass": 123.0320284,
        "inchikey14": "PVNIIMVLHYAWGP",
        "class": "Pyridine",
        "fragments": [106.0287, 78.0338, 80.0495, 51.0229],
    },
    {
        "name": "Vanillin",
        "smiles": "O=Cc1ccc(O)c(OC)c1",
        "formula": "C8H8O3",
        "neutral_mass": 152.0473441,
        "inchikey14": "WJRGINIZJRWSNW",
        "class": "Phenolic Aldehyde",
        "fragments": [151.0390, 123.0441, 109.0284, 93.0335],
    },
    {
        "name": "Resveratrol",
        "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        "formula": "C14H12O3",
        "neutral_mass": 228.0786443,
        "inchikey14": "LUKBXSAWLPMMSZ",
        "class": "Stilbenoid",
        "fragments": [181.0648, 143.0491, 131.0491, 115.0542],
    },
    {
        "name": "L-Phenylalanine",
        "smiles": "N[C@@H](Cc1ccccc1)C(=O)O",
        "formula": "C9H11NO2",
        "neutral_mass": 165.0789786,
        "inchikey14": "COLNVLDHVKWLRT",
        "class": "Amino Acid",
        "fragments": [120.0808, 91.0542, 103.0542, 77.0386],
    },
    {
        "name": "D-Glucose",
        "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "formula": "C6H12O6",
        "neutral_mass": 180.0633881,
        "inchikey14": "WQZGKKKJIJFFOK",
        "class": "Sugar",
        "fragments": [163.0601, 145.0495, 127.0390, 85.0284],
    },
    {
        "name": "Theobromine",
        "smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "formula": "C7H8N4O2",
        "neutral_mass": 180.0647256,
        "inchikey14": "YAPQBXQYLJMNSA",
        "class": "Xanthine",
        "fragments": [137.0581, 109.0632, 82.0524, 68.0243],
    },
    {
        "name": "Salicylic Acid",
        "smiles": "Oc1ccccc1C(=O)O",
        "formula": "C7H6O3",
        "neutral_mass": 138.0316941,
        "inchikey14": "YGSDEFSMJLZEOE",
        "class": "Phenolic Acid",
        "fragments": [120.0211, 92.0262, 77.0386, 65.0386],
    },
    {
        "name": "Curcumin",
        "smiles": "O=C(/C=C/c1ccc(O)c(OC)c1)CC(=O)/C=C/c1ccc(O)c(OC)c1",
        "formula": "C21H20O6",
        "neutral_mass": 368.1259884,
        "inchikey14": "VFLASGCUNZVGPO",
        "class": "Polyphenol",
        "fragments": [177.0546, 149.0597, 137.0597, 192.0781],
    },
    {
        "name": "Ferulic Acid",
        "smiles": "COc1cc(/C=C/C(=O)O)ccc1O",
        "formula": "C10H10O4",
        "neutral_mass": 194.0579088,
        "inchikey14": "KSEBMYQBYZTDHS",
        "class": "Hydroxycinnamic Acid",
        "fragments": [179.0339, 135.0441, 107.0491, 89.0386],
    },
    {
        "name": "Coumarin",
        "smiles": "O=C1Oc2ccccc2C=C1",
        "formula": "C9H6O2",
        "neutral_mass": 146.0367794,
        "inchikey14": "ZCVAOQKBIPHGAC",
        "class": "Benzopyrone",
        "fragments": [118.0413, 90.0464, 89.0386, 63.0229],
    },
    {
        "name": "Caffeic Acid",
        "smiles": "O=C(O)/C=C/c1ccc(O)c(O)c1",
        "formula": "C9H8O4",
        "neutral_mass": 180.0422587,
        "inchikey14": "QAIPRVGARKJPMS",
        "class": "Hydroxycinnamic Acid",
        "fragments": [163.0390, 135.0441, 123.0441, 89.0386],
    },
    {
        "name": "Epicatechin",
        "smiles": "Oc1cc(O)c2c(c1)O[C@@H](c1ccc(O)c(O)c1)[C@@H](O)C2",
        "formula": "C15H14O6",
        "neutral_mass": 290.0790382,
        "inchikey14": "PFTAWBLQPZVEMU",
        "class": "Flavan-3-ol",
        "fragments": [139.0390, 123.0441, 151.0390, 165.0546],
    },
    {
        "name": "Berberine",
        "smiles": "COc1ccc2c(c1OC)CC[N+]1=C2Cc2cc3c(cc21)OCO3",
        "formula": "C20H18NO4+",
        "neutral_mass": 336.1235831,
        "inchikey14": "FPYJFEZQNODWFG",
        "class": "Alkaloid",
        "fragments": [320.0917, 292.0968, 278.0812, 306.0761],
    },
    {
        "name": "Artemisinin",
        "smiles": "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C23OO4",
        "formula": "C15H22O5",
        "neutral_mass": 282.1467238,
        "inchikey14": "BLUAFEHZUOTIPD",
        "class": "Terpene Lactone",
        "fragments": [265.1434, 219.1380, 209.1172, 163.1117],
    },
]


class MockDataGenerator:
    """Synthetic generator for realistic spectra, libraries, and candidate databases."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def calculate_precursor_mz(self, neutral_mass: float, adduct: str) -> float:
        """Calculates expected precursor m/z using authoritative competition adduct deltas."""
        return chem_calc_precursor_mz(neutral_mass, adduct)

    def simulate_ms2_peaks(
        self,
        neutral_mass: float,
        precursor_mz: float,
        adduct: str,
        collision_energy: float,
        is_13c_mispick: bool = False,
        base_fragments: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates realistic MS2 peaks including precursor cluster,
        neutral loss peaks, diagnostic ions, and mass jitter.
        """
        mzs: List[float] = []
        intensities: List[float] = []

        # 1. Precursor ion residual
        ce_factor = math.exp(-0.035 * max(10.0, collision_energy))
        prec_intensity = max(5.0, 100.0 * ce_factor)
        mzs.append(precursor_mz)
        intensities.append(prec_intensity)

        # 2. If 13C mispick: add residual monoisotopic peak (M_0) in precursor cluster
        if is_13c_mispick:
            m_zero_mz = precursor_mz - CARBON_13_DELTA
            mzs.append(m_zero_mz)
            # Typically 1.5x - 2.5x higher than the M+1 mispick ion in low-energy DDA
            intensities.append(prec_intensity * 1.8)

        # 3. Base diagnostic fragments
        if base_fragments:
            for frag_mz in base_fragments:
                if frag_mz < precursor_mz - 5.0:
                    jitter = self.rng.gauss(0, 0.0015)  # ~2 ppm jitter
                    mzs.append(frag_mz + jitter)
                    intensities.append(self.rng.uniform(15.0, 95.0))

        # 4. Common neutral losses (H2O: 18.0106, CO: 27.9949, CO2: 43.9898)
        neutral_losses = [18.010565, 27.994915, 43.989829]
        for nl in neutral_losses:
            nl_mz = precursor_mz - nl
            if nl_mz > 40.0:
                jitter = self.rng.gauss(0, 0.001)
                mzs.append(nl_mz + jitter)
                intensities.append(self.rng.uniform(8.0, 45.0))

        # 5. Low-mass background chemical noise peaks
        n_noise = self.rng.randint(2, 6)
        for _ in range(n_noise):
            noise_mz = self.rng.uniform(50.0, min(200.0, precursor_mz))
            mzs.append(noise_mz)
            intensities.append(self.rng.uniform(0.5, 3.5))

        # Sort ascending by m/z
        mzs_arr = np.array(mzs, dtype=np.float64)
        ints_arr = np.array(intensities, dtype=np.float64)
        order = np.argsort(mzs_arr)
        mzs_arr = mzs_arr[order]
        ints_arr = ints_arr[order]

        # Normalize base peak to 100.0
        if len(ints_arr) > 0 and np.max(ints_arr) > 0:
            ints_arr = (ints_arr / np.max(ints_arr)) * 100.0

        return mzs_arr, ints_arr

    def generate_spectrum(
        self,
        compound_idx: Optional[int] = None,
        adduct: Optional[str] = None,
        collision_energy: Optional[float] = None,
        is_13c_mispick: bool = False,
        molecule_id: Optional[str] = None,
    ) -> SpectrumData:
        """Generates a single rich synthetic spectrum."""
        if compound_idx is None:
            compound = self.rng.choice(ARCHETYPE_COMPOUNDS)
        else:
            compound = ARCHETYPE_COMPOUNDS[compound_idx % len(ARCHETYPE_COMPOUNDS)]

        if adduct is None:
            adduct = self.rng.choice(list(COMPETITION_ADDUCTS.keys()))

        if collision_energy is None:
            collision_energy = float(self.rng.choice([15.0, 20.0, 35.0, 45.0]))

        neutral_mass = compound["neutral_mass"]
        precursor_mz = self.calculate_precursor_mz(neutral_mass, adduct)

        if is_13c_mispick:
            precursor_mz += CARBON_13_DELTA

        polarity = COMPETITION_ADDUCTS[adduct].polarity
        mzs, ints = self.simulate_ms2_peaks(
            neutral_mass=neutral_mass,
            precursor_mz=precursor_mz,
            adduct=adduct,
            collision_energy=collision_energy,
            is_13c_mispick=is_13c_mispick,
            base_fragments=compound.get("fragments"),
        )

        if molecule_id is None:
            molecule_id = f"MOL_{self.rng.randint(1000, 9999)}"

        return SpectrumData(
            molecule_id=molecule_id,
            precursor_mz=precursor_mz,
            adduct=adduct,
            polarity=polarity,
            collision_energy=collision_energy,
            mz_array=mzs,
            intensity_array=ints,
            smiles=compound["smiles"],
            inchikey14=compound["inchikey14"],
            neutral_mass=neutral_mass,
            is_13c_mispick=is_13c_mispick,
            metadata={"compound_name": compound["name"], "formula": compound["formula"]},
        )

    def generate_dataset(
        self,
        n_samples: int = 20,
        adduct_list: Optional[List[str]] = None,
        mispick_ratio: float = 0.15,
    ) -> List[SpectrumData]:
        """Generates a batch of mock query spectra."""
        if adduct_list is None:
            adduct_list = list(COMPETITION_ADDUCTS.keys())

        spectra: List[SpectrumData] = []
        for i in range(n_samples):
            compound_idx = i % len(ARCHETYPE_COMPOUNDS)
            adduct = adduct_list[i % len(adduct_list)]
            is_mispick = self.rng.random() < mispick_ratio
            mol_id = f"MOL_{i+1:04d}"
            ce = float(self.rng.choice([20.0, 35.0]))
            spec = self.generate_spectrum(
                compound_idx=compound_idx,
                adduct=adduct,
                collision_energy=ce,
                is_13c_mispick=is_mispick,
                molecule_id=mol_id,
            )
            spectra.append(spec)
        return spectra

    def generate_mock_library(
        self,
        n_entries: int = 500,
        embedding_dim: int = 512,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generates mock Tier 1 spectral reference library.
        Returns (library_df, embeddings_matrix) where embeddings are unit L2-norm float32.
        """
        records: List[Dict[str, Any]] = []
        raw_embeddings: List[np.ndarray] = []

        for i in range(n_entries):
            compound = ARCHETYPE_COMPOUNDS[i % len(ARCHETYPE_COMPOUNDS)]
            adduct = "[M+H]+" if i % 2 == 0 else "[M-H]-"
            neutral_mass = compound["neutral_mass"] + (i // len(ARCHETYPE_COMPOUNDS)) * 0.05
            prec_mz = self.calculate_precursor_mz(neutral_mass, adduct)

            records.append({
                "entry_id": f"LIB_{i+1:05d}",
                "smiles": compound["smiles"],
                "inchikey14": compound["inchikey14"],
                "neutral_mass": neutral_mass,
                "adduct": adduct,
                "precursor_mz": prec_mz,
            })

            # Random vector normalized to unit length
            vec = self.np_rng.standard_normal(embedding_dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            raw_embeddings.append(vec)

        df = pd.DataFrame(records)
        embeddings = np.vstack(raw_embeddings)
        return df, embeddings

    def generate_mock_candidate_db(
        self,
        n_candidates: int = 10000,
        n_bits: int = 4096,
    ) -> Dict[str, Any]:
        """
        Generates mock Tier 2 candidate database for popcount Tanimoto benchmarking.
        Fingerprints are bitpacked uint64 arrays of shape (n_candidates, n_bits // 64).
        """
        records: List[Dict[str, Any]] = []
        n_uint64 = n_bits // 64

        # Generate uint64 bitpacked arrays directly
        fingerprints = self.np_rng.integers(
            0, np.iinfo(np.uint64).max, size=(n_candidates, n_uint64), dtype=np.uint64
        )

        for i in range(n_candidates):
            compound = ARCHETYPE_COMPOUNDS[i % len(ARCHETYPE_COMPOUNDS)]
            mass_jitter = self.rng.uniform(-0.5, 0.5)
            neutral_mass = compound["neutral_mass"] + mass_jitter
            records.append({
                "candidate_id": f"CAND_{i+1:06d}",
                "smiles": compound["smiles"],
                "inchikey14": compound["inchikey14"],
                "neutral_mass": neutral_mass,
                "formula": compound["formula"],
            })

        df = pd.DataFrame(records)
        return {
            "candidate_df": df,
            "fingerprints": fingerprints,
            "n_bits": n_bits,
        }
