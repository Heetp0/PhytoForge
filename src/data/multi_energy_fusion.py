"""
Multi-Energy Spectral Fusion Engine (Module 1b).
Provides attention/entropy-weighted late pooling across collision energy spectra
and high-resolution peak merging/deduplication for CASMI-Omega v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class SpectralFrame:
    """
    Container for a single MS2 acquisition frame at a specific collision energy.

    Attributes:
        collision_energy: Acquisition collision energy in eV (defaults to 35.0 eV).
        peaks: (N, 2) array of [m/z, intensity].
        embedding: Optional 1D feature/representation vector (e.g. DreaMS 1024D).
        metadata: Arbitrary metadata dictionary for instrument/run parameters.
    """
    collision_energy: Optional[float] = 35.0
    peaks: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.collision_energy is not None:
            self.collision_energy = float(self.collision_energy)

        # Normalize peaks array
        if self.peaks is None:
            self.peaks = np.zeros((0, 2), dtype=np.float64)
        elif not isinstance(self.peaks, np.ndarray):
            self.peaks = np.array(self.peaks, dtype=np.float64)
        else:
            self.peaks = self.peaks.astype(np.float64, copy=False)

        if self.peaks.ndim == 1:
            if self.peaks.size == 2:
                self.peaks = self.peaks.reshape(1, 2)
            elif self.peaks.size == 0:
                self.peaks = np.zeros((0, 2), dtype=np.float64)
            else:
                raise ValueError(
                    f"1D peak array must have size 2 ([mz, intensity]), got size {self.peaks.size}"
                )
        elif self.peaks.ndim == 2:
            if self.peaks.shape[1] != 2 and self.peaks.shape[0] == 0:
                self.peaks = np.zeros((0, 2), dtype=np.float64)
            elif self.peaks.shape[1] != 2:
                raise ValueError(
                    f"peaks array must have shape (N, 2), got {self.peaks.shape}"
                )
        else:
            raise ValueError(f"peaks array must be 2-dimensional, got {self.peaks.ndim}D")

        # Normalize embedding array
        if self.embedding is not None:
            if not isinstance(self.embedding, np.ndarray):
                self.embedding = np.array(self.embedding, dtype=np.float32)
            else:
                self.embedding = self.embedding.astype(np.float32, copy=False)


def fuse_peaks(
    frames: List[SpectralFrame],
    tolerance_da: float = 0.02,
    intensity_aggregation: str = "max",
) -> np.ndarray:
    """
    Merge and deduplicate MS2 peaks across collision energies within mass tolerance.

    Parameters:
        frames: List of SpectralFrame instances.
        tolerance_da: Absolute mass window in Daltons for grouping peaks.
        intensity_aggregation: Method to combine peak intensities within a cluster:
            - 'max': Keep peak maximum across energies (preserves strongest fragmentation).
            - 'sum': Total ion intensity across energies.
            - 'mean': Average intensity across energies.

    Returns:
        (N, 2) float64 array of merged peaks [m/z, intensity] sorted by m/z ascending.
    """
    if not frames:
        return np.zeros((0, 2), dtype=np.float64)

    # Collect all valid peaks from frames
    all_peak_chunks = []
    for frame in frames:
        if frame.peaks is not None and len(frame.peaks) > 0:
            all_peak_chunks.append(frame.peaks)

    if not all_peak_chunks:
        return np.zeros((0, 2), dtype=np.float64)

    stacked = np.vstack(all_peak_chunks)
    if len(stacked) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Sort peaks by m/z ascending
    sort_idx = np.argsort(stacked[:, 0])
    sorted_peaks = stacked[sort_idx]

    # Cluster peaks within tolerance window
    merged_mzs: List[float] = []
    merged_ints: List[float] = []

    current_cluster_mzs: List[float] = [float(sorted_peaks[0, 0])]
    current_cluster_ints: List[float] = [float(sorted_peaks[0, 1])]

    for i in range(1, len(sorted_peaks)):
        mz = float(sorted_peaks[i, 0])
        intensity = float(sorted_peaks[i, 1])

        # Check tolerance relative to cluster anchor (first peak in cluster)
        if (mz - current_cluster_mzs[0]) <= tolerance_da:
            current_cluster_mzs.append(mz)
            current_cluster_ints.append(intensity)
        else:
            # Finalize current cluster
            total_int = sum(current_cluster_ints)
            if total_int > 0:
                weighted_mz = sum(m * it for m, it in zip(current_cluster_mzs, current_cluster_ints)) / total_int
            else:
                weighted_mz = sum(current_cluster_mzs) / len(current_cluster_mzs)

            if intensity_aggregation == "sum":
                agg_int = total_int
            elif intensity_aggregation == "mean":
                agg_int = total_int / len(current_cluster_ints)
            else:  # default 'max'
                agg_int = max(current_cluster_ints)

            merged_mzs.append(weighted_mz)
            merged_ints.append(agg_int)

            current_cluster_mzs = [mz]
            current_cluster_ints = [intensity]

    # Finalize last cluster
    if current_cluster_mzs:
        total_int = sum(current_cluster_ints)
        if total_int > 0:
            weighted_mz = sum(m * it for m, it in zip(current_cluster_mzs, current_cluster_ints)) / total_int
        else:
            weighted_mz = sum(current_cluster_mzs) / len(current_cluster_mzs)

        if intensity_aggregation == "sum":
            agg_int = total_int
        elif intensity_aggregation == "mean":
            agg_int = total_int / len(current_cluster_ints)
        else:
            agg_int = max(current_cluster_ints)

        merged_mzs.append(weighted_mz)
        merged_ints.append(agg_int)

    return np.column_stack([
        np.array(merged_mzs, dtype=np.float64),
        np.array(merged_ints, dtype=np.float64),
    ])


class MultiEnergyFusionEngine:
    """
    Attention/entropy-weighted late pooling across collision energy spectra.
    Prioritizes 30-40 eV core structural spectra (optimal fragmentation near 35 eV).
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        optimal_ce: float = 35.0,
        ce_decay: float = 15.0,
    ):
        self.embedding_dim = embedding_dim
        self.optimal_ce = optimal_ce
        self.ce_decay = ce_decay

    def fuse_embeddings(self, frames: List[SpectralFrame]) -> np.ndarray:
        """
        Pool multiple spectral embeddings into a single L2-normalized vector.

        Parameters:
            frames: List of SpectralFrame instances containing embedding vectors.

        Returns:
            Unit-normalized float32 numpy array of shape (embedding_dim,).
        """
        if not frames:
            raise ValueError("Cannot fuse empty list of spectral frames")

        valid_frames = [f for f in frames if f.embedding is not None]
        if not valid_frames:
            raise ValueError("No valid embeddings found in provided spectral frames")

        for f in valid_frames:
            if f.embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {f.embedding.shape[0]}"
                )

        if len(valid_frames) == 1:
            emb = valid_frames[0].embedding.copy()
            norm = float(np.linalg.norm(emb))
            if norm == 0.0 or np.isnan(norm):
                return np.zeros(self.embedding_dim, dtype=np.float32)
            return (emb / norm).astype(np.float32)

        # Weighting: prioritize 30-40 eV core structural spectra (decay from optimal_ce)
        weights = []
        for f in valid_frames:
            ce = f.collision_energy if f.collision_energy is not None else self.optimal_ce
            dist_from_optimal = abs(ce - self.optimal_ce)
            weight = np.exp(-dist_from_optimal / self.ce_decay)
            weights.append(weight)

        weights_arr = np.array(weights, dtype=np.float32)
        weight_sum = np.sum(weights_arr)
        if weight_sum > 0:
            weights_arr /= weight_sum
        else:
            weights_arr = np.ones_like(weights_arr) / len(weights_arr)

        fused = np.zeros(self.embedding_dim, dtype=np.float32)
        for w, f in zip(weights_arr, valid_frames):
            fused += w * f.embedding

        norm = float(np.linalg.norm(fused))
        if norm == 0.0 or np.isnan(norm):
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return (fused / norm).astype(np.float32)

    def fuse_peaks(
        self,
        frames: List[SpectralFrame],
        tolerance_da: float = 0.02,
        intensity_aggregation: str = "max",
    ) -> np.ndarray:
        """Merge and deduplicate MS2 peaks across collision energies."""
        return fuse_peaks(
            frames=frames,
            tolerance_da=tolerance_da,
            intensity_aggregation=intensity_aggregation,
        )

    def fuse(
        self,
        frames: List[SpectralFrame],
        tolerance_da: float = 0.02,
        intensity_aggregation: str = "max",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute full fusion across both embeddings and peak lists.

        Returns:
            Tuple of (fused_embedding: np.ndarray, fused_peaks: np.ndarray).
        """
        fused_emb = self.fuse_embeddings(frames)
        fused_p = self.fuse_peaks(
            frames=frames,
            tolerance_da=tolerance_da,
            intensity_aggregation=intensity_aggregation,
        )
        return fused_emb, fused_p

    def from_query_spectra(
        self,
        query_spectra: List[Any],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[SpectralFrame]:
        """
        Construct SpectralFrames from QuerySpectrum or SpectrumData objects.
        """
        frames: List[SpectralFrame] = []
        for i, qs in enumerate(query_spectra):
            # Extract collision energy
            ce = getattr(qs, "collision_energy", None)

            # Extract peaks
            if hasattr(qs, "ms2_peaks"):
                peaks = qs.ms2_peaks
            elif hasattr(qs, "mz_array") and hasattr(qs, "intensity_array"):
                mz = np.array(qs.mz_array, dtype=np.float64)
                it = np.array(qs.intensity_array, dtype=np.float64)
                if len(mz) > 0 and len(it) > 0:
                    peaks = np.column_stack([mz, it])
                else:
                    peaks = np.zeros((0, 2), dtype=np.float64)
            else:
                peaks = np.zeros((0, 2), dtype=np.float64)

            # Extract embedding
            emb = embeddings[i] if embeddings is not None and i < len(embeddings) else None

            frames.append(SpectralFrame(
                collision_energy=ce,
                peaks=peaks,
                embedding=emb,
            ))
        return frames
