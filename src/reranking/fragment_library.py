"""
Fragment library extraction and knapsack assembly for Phase 3 PhytoForge pipeline.

Provides:
- parse_peaks_field: Resilient parser for diverse spectrum peak representations.
- NeutralLossLibrary: Ingests training spectra, calculates accurate precursor neutral mass
  and neutral losses, and stores fragments indexed by integer millimass units
  (round(delta_m * 10000)).
- KnapsackAssembler: Combinatorial fragment assembler solving subset-sum over integer
  millimass units using Meet-in-the-Middle (itertools.combinations_with_replacement + bisect),
  strictly guarded by iteration caps and wall-clock timeouts.
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
from pathlib import Path
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.chemistry.adducts import calculate_neutral_mass


def parse_peaks_field(
    raw: Any,
    row: Optional[Union[Dict[str, Any], pd.Series]] = None,
) -> List[Tuple[float, float]]:
    """
    Resiliently parses spectrum peaks into a list of (m/z, intensity) float tuples.

    Handles:
      - 2D np.ndarray of shape (N, 2)
      - 1D np.ndarray of m/z values (defaults intensity to 100.0)
      - List/tuple of (mz, intensity) pairs or numbers
      - JSON string of pairs: '[[105.03, 10.5], [121.02, 45.2]]'
      - Semicolon-delimited tokens: '105.03:10.5;121.02:45.2' or '105.03 10.5;121.02 45.2'
      - Comma/whitespace delimited numbers
      - Separate mz_array and intensity_array columns if raw is None/NaN and row is provided
      - Malformed or corrupt data (skips invalid items gracefully)
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        # Check if row has separate mz_array and intensity_array
        if row is not None:
            mzs = None
            ints = None
            if isinstance(row, dict):
                mzs = row.get("mz_array") or row.get("mzs")
                ints = row.get("intensity_array") or row.get("intensities")
            elif isinstance(row, pd.Series):
                if "mz_array" in row:
                    mzs = row["mz_array"]
                elif "mzs" in row:
                    mzs = row["mzs"]
                if "intensity_array" in row:
                    ints = row["intensity_array"]
                elif "intensities" in row:
                    ints = row["intensities"]

            if mzs is not None:
                parsed_mzs = parse_peaks_field(mzs)
                if ints is not None:
                    parsed_ints = parse_peaks_field(ints)
                    # Merge arrays if both exist
                    result: List[Tuple[float, float]] = []
                    for idx, (m_val, _) in enumerate(parsed_mzs):
                        int_val = parsed_ints[idx][0] if idx < len(parsed_ints) else 100.0
                        result.append((m_val, int_val))
                    return result
                return parsed_mzs
        return []

    # 2D numpy array
    if isinstance(raw, np.ndarray):
        if raw.ndim == 2 and raw.shape[1] >= 2:
            pairs: List[Tuple[float, float]] = []
            for r in raw:
                try:
                    m, i = float(r[0]), float(r[1])
                    if np.isfinite(m) and np.isfinite(i):
                        pairs.append((m, i))
                except (ValueError, TypeError):
                    continue
            return pairs
        elif raw.ndim == 1:
            pairs = []
            for r in raw:
                try:
                    m = float(r)
                    if np.isfinite(m):
                        pairs.append((m, 100.0))
                except (ValueError, TypeError):
                    continue
            return pairs
        return []

    # List or tuple
    if isinstance(raw, (list, tuple)):
        pairs = []
        for item in raw:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                try:
                    m = float(item[0])
                    i = float(item[1])
                    if np.isfinite(m) and np.isfinite(i):
                        pairs.append((m, i))
                except (ValueError, TypeError):
                    continue
            elif isinstance(item, (int, float, np.number)):
                try:
                    m = float(item)
                    if np.isfinite(m):
                        pairs.append((m, 100.0))
                except (ValueError, TypeError):
                    continue
        return pairs

    if not isinstance(raw, str) or not raw.strip():
        return []

    s = raw.strip()

    # JSON formatted string
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            parsed = json.loads(s)
            return parse_peaks_field(parsed)
        except Exception:
            pass

    # Semicolon-delimited tokens: '105.03:10.5;121.02:45.2' or '105.03 10.5;121.02 45.2'
    if ";" in s or "\n" in s:
        tokens = [t.strip() for t in re.split(r"[;\n]+", s) if t.strip()]
        pairs = []
        for tok in tokens:
            parts = re.split(r"[:,\s]+", tok)
            if len(parts) >= 2:
                try:
                    m = float(parts[0])
                    i = float(parts[1])
                    if np.isfinite(m) and np.isfinite(i):
                        pairs.append((m, i))
                except (ValueError, TypeError):
                    continue
            elif len(parts) == 1 and parts[0]:
                try:
                    m = float(parts[0])
                    if np.isfinite(m):
                        pairs.append((m, 100.0))
                except (ValueError, TypeError):
                    continue
        return pairs

    # Colon-delimited pairs with whitespace: '105.03:10.5 121.02:45.2'
    if ":" in s:
        tokens = s.split()
        pairs = []
        for tok in tokens:
            parts = tok.split(":")
            if len(parts) >= 2:
                try:
                    m = float(parts[0])
                    i = float(parts[1])
                    if np.isfinite(m) and np.isfinite(i):
                        pairs.append((m, i))
                except (ValueError, TypeError):
                    continue
        return pairs

    # Simple list of numbers separated by whitespace or commas
    tokens = re.split(r"[,;\s]+", s)
    pairs = []
    for tok in tokens:
        if tok:
            try:
                m = float(tok)
                if np.isfinite(m):
                    pairs.append((m, 100.0))
            except (ValueError, TypeError):
                continue
    return pairs


class NeutralLossLibrary:
    """
    Extracts and indexes neutral losses from mass spectra datasets.

    Storage:
      - Keyed exclusively by integer millimass units: round(neutral_loss_da * 10000).
      - No float dict keys are ever used, preventing precision roundtrip errors.
      - Values are lists of candidate identifiers or SMILES strings.
    """

    def __init__(
        self,
        min_intensity: float = 0.01,
        max_mass: float = 500.0,
        library: Optional[Dict[int, List[str]]] = None,
    ) -> None:
        self.min_intensity = float(min_intensity)
        self.max_mass = float(max_mass)
        self.library: Dict[int, List[str]] = {}

        if library is not None:
            for k, v in library.items():
                if isinstance(k, (float, np.floating)):
                    raise TypeError(f"NeutralLossLibrary keys must be int, got float: {k}")
                int_k = int(k)
                self.library[int_k] = list(v)

    def __len__(self) -> int:
        return len(self.library)

    def __getitem__(self, key: int) -> List[str]:
        return self.library[int(key)]

    def __contains__(self, key: int) -> bool:
        return int(key) in self.library

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        min_intensity: Optional[float] = None,
        max_mass: Optional[float] = None,
        smiles_col: str = "smiles",
    ) -> Dict[str, int]:
        """
        Extracts neutral losses from a DataFrame and stores them in the library.

        Args:
            df: DataFrame containing precursor_mz, adduct, and peaks (or mz_array/intensity_array).
            min_intensity: Minimum relative intensity threshold (fraction of base peak).
            max_mass: Maximum neutral loss mass in Daltons.
            smiles_col: Column name containing SMILES or candidate identifiers.

        Returns:
            Dictionary with statistics:
            {"total_spectra": int, "unique_keys": int, "total_fragments": int}
        """
        if min_intensity is None:
            min_intensity = self.min_intensity
        if max_mass is None:
            max_mass = self.max_mass

        if df is None or len(df) == 0:
            return {
                "total_spectra": 0,
                "unique_keys": len(self.library),
                "total_fragments": sum(len(v) for v in self.library.values()),
            }

        cols = {str(c).lower(): c for c in df.columns}

        # Resolve column aliases
        prec_col = None
        for cand in ["precursor_mz", "precursormz", "prec_mz", "mz"]:
            if cand in cols:
                prec_col = cols[cand]
                break

        adduct_col = None
        for cand in ["adduct", "ion_type", "precursor_type"]:
            if cand in cols:
                adduct_col = cols[cand]
                break

        peaks_col = None
        for cand in ["peaks", "ms2_peaks", "peak_list", "peaks_array"]:
            if cand in cols:
                peaks_col = cols[cand]
                break

        actual_smiles_col = None
        if smiles_col in df.columns:
            actual_smiles_col = smiles_col
        else:
            for cand in ["smiles", "canonical_smiles", "structure"]:
                if cand in cols:
                    actual_smiles_col = cols[cand]
                    break

        total_processed = 0
        total_frags_added = 0

        for _, row in df.iterrows():
            total_processed += 1

            if prec_col is None or pd.isna(row[prec_col]):
                continue
            try:
                prec_mz = float(row[prec_col])
            except (ValueError, TypeError):
                continue
            if prec_mz <= 0.0 or np.isnan(prec_mz):
                continue

            if adduct_col is None or pd.isna(row[adduct_col]):
                continue
            adduct_str = str(row[adduct_col]).strip()

            try:
                prec_neutral_mass = calculate_neutral_mass(prec_mz, adduct_str)
            except Exception:
                continue
            if prec_neutral_mass <= 0.0 or np.isnan(prec_neutral_mass):
                continue

            raw_peaks = row[peaks_col] if peaks_col is not None else None
            peaks = parse_peaks_field(raw_peaks, row=row)
            if not peaks:
                continue

            # Compute relative intensity threshold
            max_int = max(intensity for mz, intensity in peaks)
            cutoff = max_int * min_intensity if max_int > 0.0 else 0.0

            # Candidate label
            label = None
            if actual_smiles_col is not None and pd.notna(row[actual_smiles_col]):
                s = str(row[actual_smiles_col]).strip()
                if s:
                    label = s
            if not label and "molecule_id" in row and pd.notna(row["molecule_id"]):
                label = str(row["molecule_id"]).strip()

            for frag_mz, intensity in peaks:
                if intensity < cutoff:
                    continue
                neutral_loss = prec_neutral_mass - frag_mz
                if neutral_loss <= 0.0 or neutral_loss > max_mass:
                    continue

                key = int(round(neutral_loss * 10000))
                if key <= 0:
                    continue

                val = label if label is not None else f"NL_{key}"
                if key not in self.library:
                    self.library[key] = []
                if val not in self.library[key]:
                    self.library[key].append(val)
                    total_frags_added += 1

        return {
            "total_spectra": total_processed,
            "unique_keys": len(self.library),
            "total_fragments": total_frags_added,
        }

    def build_from_file(
        self,
        file_path: Union[str, Path],
        min_intensity: Optional[float] = None,
        max_mass: Optional[float] = None,
        smiles_col: str = "smiles",
    ) -> Dict[str, int]:
        """Loads a CSV or Parquet file and builds the library."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        elif suffix in [".csv", ".tsv", ".txt"]:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Must be .parquet or .csv")

        return self.build_from_dataframe(
            df,
            min_intensity=min_intensity,
            max_mass=max_mass,
            smiles_col=smiles_col,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Serializes the library to disk in JSON or Pickle format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.library.items()}, f)
        else:
            with open(p, "wb") as f:
                pickle.dump(self.library, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Union[str, Path]) -> None:
        """Loads the library from disk, strictly validating that all keys are integers."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Library file not found: {p}")
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.library = {int(k): list(v) for k, v in raw.items()}
        else:
            with open(p, "rb") as f:
                raw = pickle.load(f)
            self.library = {int(k): list(v) for k, v in raw.items()}

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> NeutralLossLibrary:
        """Loads and returns a NeutralLossLibrary instance from file."""
        instance = cls()
        instance.load(path)
        return instance

    def query(self, target_mass_da: float, ppm_tolerance: float = 5.0) -> List[int]:
        """
        Queries the library for matching integer millimass keys within ppm tolerance.

        Args:
            target_mass_da: Query target mass in Daltons.
            ppm_tolerance: Tolerance in parts-per-million.

        Returns:
            Sorted list of matching integer millimass keys.
        """
        if target_mass_da <= 0.0 or not self.library:
            return []

        tol_da = target_mass_da * (ppm_tolerance / 1e6)
        min_k = int(round((target_mass_da - tol_da) * 10000))
        max_k = int(round((target_mass_da + tol_da) * 10000))

        if min_k > max_k:
            return []

        if (max_k - min_k) <= 2000:
            return [k for k in range(min_k, max_k + 1) if k in self.library]

        return [k for k in sorted(self.library.keys()) if min_k <= k <= max_k]


class KnapsackAssembler:
    """
    Meet-in-the-Middle combinatorial fragment assembler with strict governor guards.

    Solves subset-sum over integer millimass units using combinations_with_replacement
    and bisect binary search for complements. Bounded by a hard iteration cap (200k)
    and wall-clock timeout (2.0s).
    """

    def assemble(
        self,
        target_mass_da: float,
        library: Dict[int, List[str]],
        ppm_tolerance: float = 5.0,
        max_depth: int = 3,
        max_candidates: int = 25,
        timeout_s: float = 2.0,
    ) -> List[str]:
        """
        Assemble fragment combinations from library matching target mass within tolerance.

        Args:
            target_mass_da: Query monoisotopic neutral mass in Daltons.
            library: Dictionary mapping integer millimass units to fragment identifiers.
            ppm_tolerance: Tolerance in parts-per-million.
            max_depth: Maximum combination depth (1, 2, or 3).
            max_candidates: Maximum number of deduplicated candidates to return.
            timeout_s: Maximum execution wall-clock seconds before early termination.

        Returns:
            Deduplicated list of candidate strings (e.g. 'frag1', 'frag1+frag2').
        """
        if target_mass_da <= 0.0 or not library or max_candidates <= 0 or max_depth <= 0:
            return []

        start_time = time.perf_counter()
        MAX_ITER = 200_000
        iter_count = 0

        target_key = int(round(target_mass_da * 10000))
        if target_key <= 0:
            return []

        tol_da = target_mass_da * (ppm_tolerance / 1e6)
        delta_key = max(1, int(round(tol_da * 10000)))
        min_target = max(1, target_key - delta_key)
        max_target = target_key + delta_key

        # Filter strictly positive integer keys <= max_target
        sorted_keys = [
            int(k) for k in library.keys()
            if isinstance(k, (int, np.integer)) and not isinstance(k, bool) and 0 < k <= max_target
        ]
        sorted_keys.sort()

        if not sorted_keys:
            return []

        candidates: List[str] = []
        seen: Set[str] = set()

        def add_candidate(parts: List[str]) -> bool:
            """Adds candidate to output if new; returns True if max_candidates reached."""
            cand_str = "+".join(sorted(parts))
            if cand_str not in seen:
                seen.add(cand_str)
                candidates.append(cand_str)
                if len(candidates) >= max_candidates:
                    return True
            return False

        # --- Depth 1: Single fragment match ---
        if max_depth >= 1:
            idx_low = bisect.bisect_left(sorted_keys, min_target)
            idx_high = bisect.bisect_right(sorted_keys, max_target)
            for k in sorted_keys[idx_low:idx_high]:
                iter_count += 1
                entries = library.get(k, []) or [f"NL_{k}"]
                for s in entries:
                    if add_candidate([s]):
                        return candidates
                if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                    return candidates

        # --- Depth 2: Two fragments (k1 + k2 in [min_target, max_target], k1 <= k2) ---
        if max_depth >= 2 and len(candidates) < max_candidates:
            max_k1 = max_target // 2
            idx_k1_end = bisect.bisect_right(sorted_keys, max_k1)

            for i in range(idx_k1_end):
                k1 = sorted_keys[i]
                rem_min = max(k1, min_target - k1)
                rem_max = max_target - k1
                if rem_min > rem_max:
                    continue

                idx_low = bisect.bisect_left(sorted_keys, rem_min)
                idx_high = bisect.bisect_right(sorted_keys, rem_max)

                for j in range(idx_low, idx_high):
                    k2 = sorted_keys[j]
                    iter_count += 1
                    if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                        return candidates

                    list1 = library.get(k1, []) or [f"NL_{k1}"]
                    list2 = library.get(k2, []) or [f"NL_{k2}"]
                    for s1 in list1[:5]:
                        for s2 in list2[:5]:
                            iter_count += 1
                            if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                                return candidates
                            if add_candidate([s1, s2]):
                                return candidates

                if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                    return candidates

        # --- Depth 3: Three fragments (k1 + k2 + k3 in [min_target, max_target], k1 <= k2 <= k3) ---
        if max_depth >= 3 and len(candidates) < max_candidates:
            max_k_pair = max_target // 2
            pair_pool = [k for k in sorted_keys if k <= max_k_pair]
            max_k1 = max_target // 3

            for k1, k2 in itertools.combinations_with_replacement(pair_pool, 2):
                iter_count += 1
                if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                    return candidates

                if k1 > max_k1:
                    break

                pair_sum = k1 + k2
                if pair_sum + k2 > max_target:
                    continue

                rem_min = max(k2, min_target - pair_sum)
                rem_max = max_target - pair_sum
                if rem_min > rem_max:
                    continue

                idx_low = bisect.bisect_left(sorted_keys, rem_min)
                idx_high = bisect.bisect_right(sorted_keys, rem_max)

                for idx3 in range(idx_low, idx_high):
                    iter_count += 1
                    if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                        return candidates

                    k3 = sorted_keys[idx3]
                    list1 = library.get(k1, []) or [f"NL_{k1}"]
                    list2 = library.get(k2, []) or [f"NL_{k2}"]
                    list3 = library.get(k3, []) or [f"NL_{k3}"]

                    for s1 in list1[:3]:
                        for s2 in list2[:3]:
                            for s3 in list3[:3]:
                                iter_count += 1
                                if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                                    return candidates
                                if add_candidate([s1, s2, s3]):
                                    return candidates

                if iter_count >= MAX_ITER or (time.perf_counter() - start_time) >= timeout_s:
                    return candidates

        return candidates[:max_candidates]
