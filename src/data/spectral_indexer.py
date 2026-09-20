"""
Reference Spectral Vector Indexer (R2).

Provides zero-copy memory-mapped FP16 vector storage and aligned Parquet metadata
partitioned by ionization mode with mass-gated pre-filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SpectralHit:
    """A retrieved reference spectrum candidate matching precursor mass and vector similarity."""

    id: str
    smiles: str
    inchikey14: str
    precursor_mz: float
    adduct: str
    instrument: str
    cosine_score: float
    rank: int
    source_tier: str = "track1_dreams"

    def to_candidate(self) -> Any:
        """Convert to RetrievedCandidate for dreams_retrieval pipeline integration."""
        from src.retrieval.dreams_retrieval import RetrievedCandidate

        return RetrievedCandidate(
            smiles=self.smiles,
            inchikey14=self.inchikey14,
            score=self.cosine_score,
            instrument=self.instrument,
            source_tier=self.source_tier,
        )


def normalize_polarity(polarity: Any = None, adduct: Optional[str] = None) -> str:
    """Normalize polarity indicator to 'positive' or 'negative'."""
    if polarity is not None:
        p = str(polarity).strip().lower()
        if p in ("negative", "neg", "-", "-1", "negat", "negative_mode"):
            return "negative"
        if p in ("positive", "pos", "+", "1", "posit", "positive_mode"):
            return "positive"
    if adduct is not None:
        if "-" in str(adduct):
            return "negative"
        return "positive"
    return "positive"


class SpectralIndex:
    """Zero-copy memory-mapped reader for reference spectral vectors.

    Accesses normalized 1024-D FP16 embeddings stored in flat .npy files via
    zero-copy np.load(..., mmap_mode='r') paired with aligned Parquet metadata.
    """

    def __init__(self, index_dir: Union[str, Path]):
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Spectral index directory does not exist: {self.index_dir}"
            )

        self.manifest_path = self.index_dir / "manifest.json"
        self.manifest: Dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {}

        self.partitions: Dict[str, Dict[str, Any]] = {}
        self._load_partitions()

    def _load_partitions(self) -> None:
        """Load positive and negative index partitions."""
        dim = int(self.manifest.get("dim", 1024))
        for mode in ("positive", "negative"):
            part_dir = self.index_dir / mode
            emb_path = part_dir / "embeddings.npy"
            meta_path = part_dir / "metadata.parquet"

            if emb_path.exists() and meta_path.exists():
                embeddings = np.load(str(emb_path), mmap_mode="r")
                try:
                    df_meta = pd.read_parquet(meta_path)
                    if embeddings.shape[0] != len(df_meta):
                        raise ValueError(
                            f"Row count mismatch in {mode} partition: "
                            f"{embeddings.shape[0]} vectors vs {len(df_meta)} metadata rows."
                        )
                except Exception:
                    if hasattr(embeddings, "_mmap") and embeddings._mmap is not None:
                        try:
                            embeddings._mmap.close()
                        except Exception:
                            pass
                    self.close()
                    raise

                if "precursor_mz" in df_meta.columns:
                    masses = df_meta["precursor_mz"].to_numpy(dtype=np.float64)
                else:
                    masses = np.zeros(len(df_meta), dtype=np.float64)

                is_sorted = bool(len(masses) <= 1 or np.all(np.diff(masses) >= 0))

                self.partitions[mode] = {
                    "embeddings": embeddings,
                    "metadata": df_meta,
                    "precursor_mz": masses,
                    "is_sorted": is_sorted,
                    "count": len(df_meta),
                }
            else:
                # Initialize empty partition
                self.partitions[mode] = {
                    "embeddings": np.zeros((0, dim), dtype=np.float16),
                    "metadata": pd.DataFrame(
                        columns=[
                            "id",
                            "smiles",
                            "inchikey14",
                            "precursor_mz",
                            "adduct",
                            "instrument",
                        ]
                    ),
                    "precursor_mz": np.zeros(0, dtype=np.float64),
                    "is_sorted": True,
                    "count": 0,
                }

    def search(
        self,
        query_emb: np.ndarray,
        precursor_mz: float,
        polarity: str = "positive",
        ppm_tolerance: float = 10.0,
        top_k: int = 25,
        tolerance_ppm: Optional[float] = None,
        **kwargs: Any,
    ) -> List[SpectralHit]:
        """Perform fast mass-gated cosine vector similarity search.

        Filters candidates by precursor mass window (+- tolerance ppm) in O(log N)
        using contiguous memory-mapped slice reads, then computes normalized cosine similarity.
        """
        tol_ppm = float(
            tolerance_ppm if tolerance_ppm is not None else ppm_tolerance
        )
        pol = normalize_polarity(
            polarity, adduct=kwargs.get("adduct")
        )
        part = self.partitions.get(pol)

        if part is None or part["count"] == 0:
            return []

        if precursor_mz <= 0:
            return []

        masses = part["precursor_mz"]
        tol_da = float(precursor_mz) * (tol_ppm * 1e-6)
        min_mz = float(precursor_mz) - tol_da
        max_mz = float(precursor_mz) + tol_da

        # Binary search for mass window bounds
        if part["is_sorted"]:
            left = int(np.searchsorted(masses, min_mz - 1e-11, side="left"))
            right = int(np.searchsorted(masses, max_mz + 1e-11, side="right"))
            if left >= right:
                return []

            # Verify exact tolerance window in case of precision edge
            slice_masses = masses[left:right]
            exact_mask = np.abs(slice_masses - precursor_mz) <= (tol_da + 1e-9)
            if not np.all(exact_mask):
                valid_offsets = np.where(exact_mask)[0]
                if len(valid_offsets) == 0:
                    return []
                candidate_indices = left + valid_offsets
                cand_embs = part["embeddings"][candidate_indices]
            else:
                candidate_indices = np.arange(left, right)
                cand_embs = part["embeddings"][left:right]
        else:
            in_window = np.where((masses >= min_mz) & (masses <= max_mz))[0]
            if len(in_window) == 0:
                return []
            candidate_indices = in_window
            cand_embs = part["embeddings"][in_window]

        if len(candidate_indices) == 0:
            return []

        # Cosine similarity computation
        q = np.asarray(query_emb, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-12 or not np.isfinite(q_norm):
            scores = np.zeros(len(candidate_indices), dtype=np.float32)
        else:
            q_unit = q / q_norm
            cand_f32 = cand_embs.astype(np.float32)
            cand_norms = np.linalg.norm(cand_f32, axis=1)
            cand_norms = np.where(
                (cand_norms > 1e-12) & (np.isfinite(cand_norms)), cand_norms, 1.0
            )
            scores = np.dot(cand_f32, q_unit) / cand_norms
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            scores = np.clip(scores, 0.0, 1.0)

        n_hits = min(int(top_k), len(scores))
        if n_hits <= 0:
            return []

        top_local_idx = np.argsort(-scores)[:n_hits]
        top_global_idx = candidate_indices[top_local_idx]
        top_scores = scores[top_local_idx]

        sub_meta = part["metadata"].iloc[top_global_idx]
        records = sub_meta.to_dict(orient="records")

        hits: List[SpectralHit] = []
        for rank, (rec, score) in enumerate(zip(records, top_scores)):
            inst = str(rec.get("instrument") or "generic")
            if inst in ("nan", "None", ""):
                inst = "generic"
            hits.append(
                SpectralHit(
                    id=str(rec.get("id", f"SPEC_{top_global_idx[rank]}")),
                    smiles=str(rec.get("smiles", "")),
                    inchikey14=str(rec.get("inchikey14", "")),
                    precursor_mz=float(rec.get("precursor_mz", 0.0)),
                    adduct=str(rec.get("adduct", "")),
                    instrument=inst,
                    cosine_score=float(score),
                    rank=rank + 1,
                    source_tier="track1_dreams",
                )
            )

        return hits

    def get_embeddings(self, polarity: str = "positive") -> np.ndarray:
        """Access the underlying memory-mapped embedding array for a partition."""
        pol = normalize_polarity(polarity)
        return self.partitions[pol]["embeddings"]

    def get_metadata(self, polarity: str = "positive") -> pd.DataFrame:
        """Access the aligned metadata DataFrame for a partition."""
        pol = normalize_polarity(polarity)
        return self.partitions[pol]["metadata"]

    def close(self) -> None:
        """Safely close open memory map handles."""
        for part in self.partitions.values():
            embs = part.get("embeddings")
            if embs is not None and hasattr(embs, "_mmap") and embs._mmap is not None:
                try:
                    embs._mmap.close()
                except Exception:
                    pass

    def __enter__(self) -> "SpectralIndex":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return sum(part.get("count", 0) for part in self.partitions.values())

    def __repr__(self) -> str:
        pos_cnt = self.partitions.get("positive", {}).get("count", 0)
        neg_cnt = self.partitions.get("negative", {}).get("count", 0)
        return (
            f"SpectralIndex(dir='{self.index_dir}', "
            f"positive={pos_cnt}, negative={neg_cnt}, total={len(self)})"
        )


class SpectralIndexBuilder:
    """Builder for reference spectral vector indexes."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @classmethod
    def build_index(
        cls,
        *args: Any,
        output_dir: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        spectra: Optional[Union[pd.DataFrame, str, Path]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[Union[pd.DataFrame, dict, list]] = None,
        embedding_col: str = "embedding",
        dim: int = 1024,
        **kwargs: Any,
    ) -> Path:
        """Build and persist a partitioned, precursor-sorted spectral index.

        Supports flexible calling signatures:
        - build_index(output_dir, df=...)
        - build_index(df, output_dir)
        - build_index(output_dir, embeddings=..., metadata=...)
        - build_index(output_dir, spectra=path_to_parquet_or_csv)
        """
        # Resolve positional arguments
        if len(args) == 1:
            if output_dir is None:
                if isinstance(args[0], (pd.DataFrame, list, dict)):
                    df = args[0]
                else:
                    output_dir = args[0]
            elif df is None and spectra is None and embeddings is None:
                if isinstance(args[0], (pd.DataFrame, list, dict)):
                    df = args[0]
                else:
                    spectra = args[0]
        elif len(args) >= 2:
            a0, a1 = args[0], args[1]
            if isinstance(a0, (pd.DataFrame, list, dict)):
                df, output_dir = a0, a1
            elif isinstance(a1, (pd.DataFrame, list, dict)):
                output_dir, df = a0, a1
            else:
                p0, p1 = Path(a0), Path(a1)
                if p0.is_file() or p0.suffix in (".parquet", ".pq", ".csv", ".tsv"):
                    spectra, output_dir = p0, p1
                elif p1.is_file() or p1.suffix in (".parquet", ".pq", ".csv", ".tsv"):
                    output_dir, spectra = p0, p1
                else:
                    output_dir, spectra = p0, p1

        if output_dir is None:
            raise ValueError("output_dir must be specified")

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Assemble DataFrame
        if df is None:
            if embeddings is not None:
                embs_arr = np.asarray(embeddings)
                if metadata is not None:
                    if isinstance(metadata, pd.DataFrame):
                        df = metadata.copy()
                    else:
                        df = pd.DataFrame(metadata)
                else:
                    df = pd.DataFrame(
                        {"id": [f"SPEC_{i:06d}" for i in range(len(embs_arr))]}
                    )
                df[embedding_col] = list(embs_arr)
            elif spectra is not None:
                if isinstance(spectra, pd.DataFrame):
                    df = spectra.copy()
                else:
                    sp = Path(spectra)
                    if sp.suffix in (".parquet", ".pq"):
                        df = pd.read_parquet(sp)
                    elif sp.suffix in (".csv", ".tsv"):
                        df = pd.read_csv(
                            sp, sep="\t" if sp.suffix == ".tsv" else ","
                        )
                    else:
                        raise ValueError(f"Unsupported file format: {sp}")
            else:
                raise ValueError("No input data provided to build_index")
        else:
            if isinstance(df, (list, dict)):
                df = pd.DataFrame(df)
            else:
                df = df.copy()

        # If embeddings provided separately alongside df
        if embeddings is not None and embedding_col not in df.columns:
            df[embedding_col] = list(np.asarray(embeddings))

        # Detect embedding column if name differs
        if embedding_col not in df.columns:
            candidates = [
                "embeddings",
                "vector",
                "vectors",
                "spectral_embedding",
                "emb",
            ]
            for cand in candidates:
                if cand in df.columns:
                    embedding_col = cand
                    break

        if embedding_col not in df.columns:
            raise KeyError(
                f"Embedding column '{embedding_col}' not found in DataFrame columns: {list(df.columns)}"
            )

        # Standardize columns
        if "id" not in df.columns:
            for alt_id in ("spectrum_id", "molecule_id", "compound_id", "spec_id"):
                if alt_id in df.columns:
                    df["id"] = df[alt_id]
                    break
            else:
                df["id"] = [f"SPEC_{i:06d}" for i in range(len(df))]
        df["id"] = df["id"].astype(str)

        if "smiles" not in df.columns:
            for alt_smi in ("SMILES", "canonical_smiles"):
                if alt_smi in df.columns:
                    df["smiles"] = df[alt_smi]
                    break
            else:
                df["smiles"] = ""
        df["smiles"] = df["smiles"].fillna("").astype(str)

        if "inchikey14" not in df.columns:
            for alt_ik in ("InChIKey14", "inchikey", "ik14", "inchikey_14"):
                if alt_ik in df.columns:
                    df["inchikey14"] = df[alt_ik]
                    break
            else:
                # Compute or generate 14-char key
                keys = []
                for s in df["smiles"]:
                    if s:
                        try:
                            from rdkit import Chem

                            mol = Chem.MolFromSmiles(s)
                            if mol is not None:
                                keys.append(Chem.MolToInchiKey(mol)[:14])
                                continue
                        except Exception:
                            pass
                        keys.append(
                            hashlib.sha256(s.encode("utf-8")).hexdigest()[:14].upper()
                        )
                    else:
                        keys.append("IK14UNKNOWN000")
                df["inchikey14"] = keys
        df["inchikey14"] = df["inchikey14"].fillna("").astype(str).str[:14]

        if "precursor_mz" not in df.columns:
            for alt_mz in (
                "precursor",
                "mz",
                "mass",
                "exact_mass",
                "pepmass",
                "precursor_mass",
            ):
                if alt_mz in df.columns:
                    df["precursor_mz"] = df[alt_mz]
                    break
            else:
                df["precursor_mz"] = 0.0
        df["precursor_mz"] = pd.to_numeric(
            df["precursor_mz"], errors="coerce"
        ).fillna(0.0).astype(np.float64)

        if "adduct" not in df.columns:
            for alt_adduct in ("adduct_type", "ion"):
                if alt_adduct in df.columns:
                    df["adduct"] = df[alt_adduct]
                    break
            else:
                df["adduct"] = ""
        df["adduct"] = df["adduct"].fillna("").astype(str)

        if "instrument" not in df.columns:
            for alt_inst in ("instrument_type", "source_instrument"):
                if alt_inst in df.columns:
                    df["instrument"] = df[alt_inst]
                    break
            else:
                df["instrument"] = "generic"
        df["instrument"] = df["instrument"].fillna("generic").astype(str)

        # Polarity identification
        if "polarity" not in df.columns:
            for alt_pol in ("ion_mode", "mode", "charge"):
                if alt_pol in df.columns:
                    df["polarity"] = df[alt_pol]
                    break
            else:
                # Infer from adduct
                df["polarity"] = [
                    "negative" if "-" in add else "positive" for add in df["adduct"]
                ]

        counts: Dict[str, int] = {}
        actual_dim = dim

        for pol_key in ("positive", "negative"):
            part_dir = target_dir / pol_key
            part_dir.mkdir(parents=True, exist_ok=True)

            if pol_key == "positive":
                mask = (
                    df["polarity"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.contains(r"pos|\+|^1$|^positive_mode$")
                )
            else:
                mask = (
                    df["polarity"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.contains(r"neg|\-|^\\-1$|^negative_mode$")
                )

            sub_df = df[mask].copy()

            # Set default adduct if missing
            default_adduct = "[M+H]+" if pol_key == "positive" else "[M-H]-"
            sub_df["adduct"] = sub_df["adduct"].replace("", default_adduct)

            if sub_df.empty:
                np.save(
                    str(part_dir / "embeddings.npy"),
                    np.zeros((0, actual_dim), dtype=np.float16),
                )
                pd.DataFrame(
                    columns=[
                        "id",
                        "smiles",
                        "inchikey14",
                        "precursor_mz",
                        "adduct",
                        "instrument",
                    ]
                ).to_parquet(str(part_dir / "metadata.parquet"), index=False)
                counts[pol_key] = 0
                continue

            # Sort strictly by precursor_mz for O(log N) contiguous slicing
            sub_df = sub_df.sort_values("precursor_mz").reset_index(drop=True)

            # Extract embeddings matrix
            raw_embs = sub_df[embedding_col].values
            if (
                len(raw_embs) > 0
                and isinstance(raw_embs[0], (list, tuple, np.ndarray))
            ):
                embs_mat = np.vstack(raw_embs).astype(np.float32)
            else:
                embs_mat = np.array(raw_embs.tolist(), dtype=np.float32)

            if embs_mat.ndim == 1:
                embs_mat = embs_mat.reshape(len(sub_df), -1)

            actual_dim = embs_mat.shape[1]

            # Unit L2 normalize
            norms = np.linalg.norm(embs_mat, axis=1, keepdims=True)
            norms = np.where((norms > 1e-12) & (np.isfinite(norms)), norms, 1.0)
            embs_normed = (embs_mat / norms).astype(np.float16)

            # Persist flat .npy
            np.save(str(part_dir / "embeddings.npy"), embs_normed)

            # Persist aligned metadata Parquet
            meta_cols = [
                "id",
                "smiles",
                "inchikey14",
                "precursor_mz",
                "adduct",
                "instrument",
            ]
            sub_df[meta_cols].to_parquet(
                str(part_dir / "metadata.parquet"), index=False
            )
            counts[pol_key] = len(sub_df)

        manifest = {
            "version": "1.0.0",
            "dim": actual_dim,
            "counts": counts,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return target_dir


# Aliases for interface flexibility
SpectralIndexer = SpectralIndexBuilder
build_spectral_index = SpectralIndexBuilder.build_index
