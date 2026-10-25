"""
Reference Spectral Matcher for authentic training spectra library.

Matches query precursor m/z and MS/MS peak patterns against reference spectra
from train.parquet using fast vectorized cosine similarity and binary search indexing.
Zero external C-extensions required: pure NumPy and PyArrow implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def fast_spectral_cosine(
    peaks_query: np.ndarray,
    peaks_ref: np.ndarray,
    tolerance_da: float = 0.03,
    max_peaks: int = 30,
) -> float:
    """
    Compute square-root normalized cosine similarity between query and reference peak lists.
    Truncates to top `max_peaks` by intensity for bounded O(K^2) execution latency.
    """
    if peaks_query is None or peaks_ref is None or len(peaks_query) == 0 or len(peaks_ref) == 0:
        return 0.0

    p_a = peaks_query
    p_b = peaks_ref

    if len(p_a) > max_peaks:
        top_idx_a = np.argsort(p_a[:, 1])[::-1][:max_peaks]
        p_a = p_a[top_idx_a]
    if len(p_b) > max_peaks:
        top_idx_b = np.argsort(p_b[:, 1])[::-1][:max_peaks]
        p_b = p_b[top_idx_b]

    w_a = np.sqrt(np.maximum(p_a[:, 1], 0.0))
    w_b = np.sqrt(np.maximum(p_b[:, 1], 0.0))
    norm_a = float(np.linalg.norm(w_a))
    norm_b = float(np.linalg.norm(w_b))

    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0

    w_a /= norm_a
    w_b /= norm_b

    score = 0.0
    mzs_b = p_b[:, 0]
    for mz_i, weight_i in zip(p_a[:, 0], w_a):
        diffs = np.abs(mzs_b - mz_i)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= tolerance_da:
            score += weight_i * w_b[min_idx]

    return float(np.clip(score, 0.0, 1.0))


class ReferenceSpectralMatcher:
    """
    Precursor-indexed spectral cosine retrieval engine over authentic reference libraries.
    """

    def __init__(
        self,
        train_path: Optional[Union[str, Path]] = None,
        max_row_groups: Optional[int] = None,
    ):
        self.train_path = Path(train_path) if train_path is not None else None
        self.is_loaded: bool = False
        self.ref_mzs: np.ndarray = np.zeros(0, dtype=np.float64)
        self.ref_adducts: np.ndarray = np.array([], dtype=object)
        self.ref_iks: np.ndarray = np.array([], dtype=object)
        self.ref_smiles: np.ndarray = np.array([], dtype=object)
        self.ref_mzs_list: List[Optional[List[float]]] = []
        self.ref_ints_list: List[Optional[List[float]]] = []

        if self.train_path is not None and self.train_path.exists():
            self._load_reference_data(max_row_groups=max_row_groups)

    def is_available(self) -> bool:
        """Return True if reference library data is indexed in memory."""
        return self.is_loaded and len(self.ref_mzs) > 0

    def _load_reference_data(self, max_row_groups: Optional[int] = None) -> None:
        """Load and sort reference spectra by precursor_mz for binary range search."""
        try:
            import pyarrow.parquet as pq
            import pandas as pd

            pf = pq.ParquetFile(str(self.train_path))
            cols = [
                "inchikey14",
                "normalized_smiles",
                "precursor_mz",
                "adduct",
                "ms2_mzs",
                "ms2_normalized_intensities",
            ]

            num_groups = pf.num_row_groups
            if max_row_groups is not None:
                num_groups = min(num_groups, max_row_groups)

            df_list = [pf.read_row_group(i, columns=cols).to_pandas() for i in range(num_groups)]
            train_df = pd.concat(df_list, ignore_index=True)

            # Sort by precursor_mz for O(log N) searchsorted
            train_df = train_df.sort_values("precursor_mz").reset_index(drop=True)
            self.ref_mzs = train_df["precursor_mz"].values.astype(np.float64)
            self.ref_adducts = train_df["adduct"].values
            self.ref_iks = train_df["inchikey14"].values
            self.ref_smiles = train_df["normalized_smiles"].values
            self.ref_mzs_list = train_df["ms2_mzs"].tolist()
            self.ref_ints_list = train_df["ms2_normalized_intensities"].tolist()
            self.is_loaded = True
            logger.info("Loaded ReferenceSpectralMatcher with %d reference spectra.", len(self.ref_mzs))
        except Exception as exc:
            logger.warning("Failed to load reference library from %s: %s", self.train_path, exc)
            self.is_loaded = False

    def match_query(
        self,
        precursor_mz: float,
        peaks: Optional[np.ndarray],
        adduct: Optional[str] = None,
        tolerance_ppm: float = 10.0,
        min_cosine: float = 0.50,
        max_results: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve reference candidates matching precursor m/z within tolerance and exceeding min_cosine.
        Prioritizes matching adducts.
        """
        if not self.is_available() or peaks is None or len(peaks) == 0 or precursor_mz <= 0.0:
            return []

        delta_mz = precursor_mz * (tolerance_ppm * 1e-6)
        low_idx = int(np.searchsorted(self.ref_mzs, precursor_mz - delta_mz, side="left"))
        high_idx = int(np.searchsorted(self.ref_mzs, precursor_mz + delta_mz, side="right"))

        if high_idx <= low_idx:
            return []

        # Partition into matching adduct vs other adducts
        matching_adduct_indices = []
        other_indices = []
        for idx in range(low_idx, high_idx):
            if adduct and self.ref_adducts[idx] == adduct:
                matching_adduct_indices.append(idx)
            else:
                other_indices.append(idx)

        eval_indices = matching_adduct_indices + other_indices

        candidates: List[Dict[str, Any]] = []
        seen_iks: set = set()

        for c_idx in eval_indices:
            r_mzs = self.ref_mzs_list[c_idx]
            r_ints = self.ref_ints_list[c_idx]
            if r_mzs is None or len(r_mzs) == 0:
                continue

            r_peaks = np.column_stack([np.asarray(r_mzs, dtype=np.float64), np.asarray(r_ints, dtype=np.float64)])
            cos = fast_spectral_cosine(peaks, r_peaks, tolerance_da=0.03, max_peaks=30)

            if cos >= min_cosine:
                ik14 = str(self.ref_iks[c_idx])
                smi = str(self.ref_smiles[c_idx])
                if ik14 not in seen_iks:
                    seen_iks.add(ik14)
                    candidates.append({
                        "inchikey14": ik14,
                        "smiles": smi,
                        "score": cos,
                        "cosine": cos,
                        "source": "track1_dreams",
                        "adduct": str(self.ref_adducts[c_idx]),
                    })
                    if cos >= 0.95 and len(candidates) >= max_results:
                        break

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:max_results]
