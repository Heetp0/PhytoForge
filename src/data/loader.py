"""
Data loader and serialization module for CASMI 2026.
Provides high-performance, schema-resilient ingestion of Parquet and CSV
spectrum tables with automatic normalization, filtering, and query conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class QuerySpectrum:
    """Interface contract for query spectrum."""
    molecule_id: str
    precursor_mz: float
    adduct: str
    polarity: str
    collision_energy: Optional[float]
    mz_array: List[float]
    intensity_array: List[float]


@dataclass
class SpectrumData:
    """Rich internal representation of a mass spectrometry record."""
    molecule_id: str
    precursor_mz: float
    adduct: str
    polarity: str
    collision_energy: Optional[float]
    mz_array: np.ndarray
    intensity_array: np.ndarray
    smiles: Optional[str] = None
    inchikey14: Optional[str] = None
    neutral_mass: Optional[float] = None
    is_13c_mispick: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (self.precursor_mz > 0) or np.isnan(self.precursor_mz):
            raise ValueError(f"precursor_mz must be strictly positive, got {self.precursor_mz}")
        if not isinstance(self.mz_array, np.ndarray):
            self.mz_array = np.array(self.mz_array, dtype=np.float64)
        if not isinstance(self.intensity_array, np.ndarray):
            self.intensity_array = np.array(self.intensity_array, dtype=np.float64)
        if len(self.mz_array) != len(self.intensity_array):
            raise ValueError(
                f"Peak array length mismatch: mz_array has {len(self.mz_array)} "
                f"elements, intensity_array has {len(self.intensity_array)}"
            )

    @property
    def ms2_peaks(self) -> np.ndarray:
        """Returns (N, 2) float64 array of [mz, intensity]."""
        if len(self.mz_array) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack([self.mz_array, self.intensity_array])

    def to_query_spectrum(self) -> QuerySpectrum:
        return QuerySpectrum(
            molecule_id=str(self.molecule_id),
            precursor_mz=float(self.precursor_mz),
            adduct=str(self.adduct),
            polarity=str(self.polarity),
            collision_energy=float(self.collision_energy) if self.collision_energy is not None else None,
            mz_array=self.mz_array.tolist(),
            intensity_array=self.intensity_array.tolist(),
        )


def parse_peak_array(raw: Any) -> np.ndarray:
    """
    Resiliently parses diverse string or collection formats into a 1D float64 numpy array.
    Handles:
      - Existing np.ndarray / list / tuple
      - JSON lists: '[105.03, 121.02, 163.04]'
      - Delimited strings: '105.03 121.02' or '105.03;121.02' or '105.03,121.02'
      - Empty / None / NaN inputs
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros(0, dtype=np.float64)
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float64)
    if isinstance(raw, (list, tuple)):
        return np.array(raw, dtype=np.float64)
    if not isinstance(raw, str) or not raw.strip():
        return np.zeros(0, dtype=np.float64)

    s = raw.strip()
    # Fast-path JSON list
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            return np.array(parsed, dtype=np.float64)
        except Exception:
            s = s[1:-1]

    # Split by any sequence of whitespace, comma, or semicolon
    tokens = re.split(r"[,;\s]+", s.strip())
    vals = [float(t) for t in tokens if t]
    return np.array(vals, dtype=np.float64)


def filter_and_normalize_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    min_rel_intensity: float = 0.001,
    max_peaks: int = 100,
    target_base_intensity: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cleans, filters, and normalizes MS2 peaks:
      1. Removes peaks with intensity < min_rel_intensity * base_intensity.
      2. Retains top max_peaks by intensity.
      3. Sorts retained peaks ascending by m/z.
      4. Normalizes base peak intensity to target_base_intensity.
    """
    if len(mz_array) == 0 or len(intensity_array) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Filter out non-finite (NaN, Inf) values
    valid_mask = np.isfinite(mz_array) & np.isfinite(intensity_array) & (intensity_array > 0)
    mz_array = mz_array[valid_mask]
    intensity_array = intensity_array[valid_mask]

    if len(mz_array) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    max_int = float(np.max(intensity_array))
    if max_int <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # 1. Filter low-intensity noise
    cutoff = max_int * min_rel_intensity
    mask = intensity_array >= cutoff
    mzs = mz_array[mask]
    ints = intensity_array[mask]

    if len(mzs) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # 2. Retain top max_peaks by intensity
    if len(mzs) > max_peaks:
        top_indices = np.argsort(ints)[-max_peaks:]
        mzs = mzs[top_indices]
        ints = ints[top_indices]

    # 3. Sort ascending by m/z
    sort_order = np.argsort(mzs)
    mzs = mzs[sort_order]
    ints = ints[sort_order]

    # 4. Normalize
    ints = (ints / np.max(ints)) * target_base_intensity

    return mzs.astype(np.float64), ints.astype(np.float64)


class SpectrumLoader:
    """Unified loader and exporter for CASMI spectrum datasets."""

    @staticmethod
    def _normalize_row_columns(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves common column alias variations across datasets."""
        aliases = {
            "molecule_id": ["molecule_id", "id", "spectrum_id", "spec_id", "ID"],
            "precursor_mz": ["precursor_mz", "precursorMz", "PrecursorMZ", "prec_mz", "mz"],
            "adduct": ["adduct", "Adduct", "precursor_type", "ion_type"],
            "polarity": ["polarity", "IonMode", "ion_mode", "mode"],
            "collision_energy": ["collision_energy", "ce", "CE", "CollisionEnergy"],
            "mz_array": ["mz_array", "mzs", "mz", "peaks_mz"],
            "intensity_array": ["intensity_array", "intensities", "intensity", "peaks_intensity"],
            "smiles": ["smiles", "SMILES", "canonical_smiles"],
            "inchikey14": ["inchikey14", "inchikey", "InChIKey14", "InChIKey"],
            "neutral_mass": ["neutral_mass", "exact_mass", "monoisotopic_mass", "mass"],
        }

        normalized = {}
        for canonical, key_aliases in aliases.items():
            for alias in key_aliases:
                if alias in row_dict and row_dict[alias] is not None:
                    normalized[canonical] = row_dict[alias]
                    break

        return normalized

    @classmethod
    def from_row_dict(
        cls,
        row: Dict[str, Any],
        filter_peaks: bool = True,
        max_peaks: int = 100,
    ) -> SpectrumData:
        """Parses a dictionary row into a SpectrumData instance."""
        data = cls._normalize_row_columns(row)

        mol_id = str(data.get("molecule_id", "UNKNOWN_ID"))
        prec_mz = float(data.get("precursor_mz", 100.0))
        adduct = str(data.get("adduct", "[M+H]+"))
        
        # Normalize polarity string
        pol_raw = data.get("polarity")
        if pol_raw is not None and not pd.isna(pol_raw):
            pol_str = str(pol_raw).lower().strip()
            if pol_str in ["+", "1", "pos", "positive"]:
                polarity = "positive"
            elif pol_str in ["-", "-1", "neg", "negative"]:
                polarity = "negative"
            else:
                polarity = "positive" if "+" in adduct else "negative"
        else:
            polarity = "positive" if "+" in adduct else "negative"

        ce_raw = data.get("collision_energy")
        ce = float(ce_raw) if ce_raw is not None and not pd.isna(ce_raw) else None

        mzs = parse_peak_array(data.get("mz_array"))
        ints = parse_peak_array(data.get("intensity_array"))

        if filter_peaks:
            mzs, ints = filter_and_normalize_peaks(mzs, ints, max_peaks=max_peaks)

        smiles = data.get("smiles")
        ik14_raw = data.get("inchikey14")
        if ik14_raw is not None and not pd.isna(ik14_raw):
            ik14 = str(ik14_raw).strip()
            if len(ik14) > 14:
                ik14 = ik14.split("-")[0][:14]
        else:
            ik14 = None

        nm_raw = data.get("neutral_mass")
        neutral_mass = float(nm_raw) if nm_raw is not None and not pd.isna(nm_raw) else None

        return SpectrumData(
            molecule_id=mol_id,
            precursor_mz=prec_mz,
            adduct=adduct,
            polarity=polarity,
            collision_energy=ce,
            mz_array=mzs,
            intensity_array=ints,
            smiles=str(smiles) if smiles and not pd.isna(smiles) else None,
            inchikey14=str(ik14) if ik14 and not pd.isna(ik14) else None,
            neutral_mass=neutral_mass,
        )

    @classmethod
    def load_parquet(
        cls,
        file_path: Union[str, Path],
        max_rows: Optional[int] = None,
        filter_peaks: bool = True,
    ) -> List[SpectrumData]:
        """Loads spectrum records from Parquet using pyarrow."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        table = pq.read_table(path)
        if max_rows is not None and max_rows < table.num_rows:
            table = table.slice(0, max_rows)

        df = table.to_pandas()
        return [cls.from_row_dict(row, filter_peaks=filter_peaks) for row in df.to_dict(orient="records")]

    @classmethod
    def load_csv(
        cls,
        file_path: Union[str, Path],
        max_rows: Optional[int] = None,
        filter_peaks: bool = True,
    ) -> List[SpectrumData]:
        """Loads spectrum records from CSV file with resilient string array parsing."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, nrows=max_rows)
        return [cls.from_row_dict(row, filter_peaks=filter_peaks) for row in df.to_dict(orient="records")]

    @classmethod
    def load_file(
        cls,
        file_path: Union[str, Path],
        max_rows: Optional[int] = None,
        filter_peaks: bool = True,
    ) -> List[SpectrumData]:
        """Auto-detects file format by extension and parses spectra."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in [".parquet", ".pq"]:
            return cls.load_parquet(path, max_rows=max_rows, filter_peaks=filter_peaks)
        elif suffix in [".csv", ".tsv", ".txt"]:
            return cls.load_csv(path, max_rows=max_rows, filter_peaks=filter_peaks)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Must be .parquet or .csv")

    @staticmethod
    def to_dataframe(spectra: List[SpectrumData]) -> pd.DataFrame:
        """Converts a list of SpectrumData objects to a pandas DataFrame."""
        records = []
        for spec in spectra:
            records.append({
                "molecule_id": spec.molecule_id,
                "precursor_mz": spec.precursor_mz,
                "adduct": spec.adduct,
                "polarity": spec.polarity,
                "collision_energy": spec.collision_energy,
                "mz_array": spec.mz_array.tolist(),
                "intensity_array": spec.intensity_array.tolist(),
                "smiles": spec.smiles,
                "inchikey14": spec.inchikey14,
                "neutral_mass": spec.neutral_mass,
                "is_13c_mispick": spec.is_13c_mispick,
            })
        return pd.DataFrame(records)

    @classmethod
    def to_query_spectra(cls, spectra: List[SpectrumData]) -> List[QuerySpectrum]:
        """Converts a list of SpectrumData objects to QuerySpectrum dataclasses."""
        return [s.to_query_spectrum() for s in spectra]

    @classmethod
    def save_parquet(cls, spectra: List[SpectrumData], file_path: Union[str, Path]) -> Path:
        """Serializes spectra list to Parquet with pyarrow list<float64> types."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = cls.to_dataframe(spectra)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        return path

    @classmethod
    def save_csv(cls, spectra: List[SpectrumData], file_path: Union[str, Path]) -> Path:
        """Serializes spectra list to CSV with JSON-encoded arrays."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = cls.to_dataframe(spectra)
        # Serialize lists to JSON string for CSV compatibility
        df["mz_array"] = df["mz_array"].apply(json.dumps)
        df["intensity_array"] = df["intensity_array"].apply(json.dumps)
        df.to_csv(path, index=False)
        return path
