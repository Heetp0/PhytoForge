"""Synthetic Ranker Dataset Generator for LightGBM LambdaMART training.

Generates 33-dimensional multimodal feature matrices, integer relevance labels,
and query group boundaries for listwise learning-to-rank models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from src.chemistry.adducts import calculate_neutral_mass
from src.pipeline import _compute_candidate_fragment_match_ratio
from src.reranking.botanical_substructures import BotanicalSubstructureScorer
from src.reranking.meta_ranker import GBDTMetaRanker
from src.retrieval.database_search import DBCandidate, SoftDatabaseSearcher
from src.retrieval.mist_formula_router import MISTFormulaRouter


class RankerDatasetGenerator:
    """Generates synthetic multimodal feature matrices and listwise ranking labels."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_synthetic_ranking_dataset(
        self,
        n_queries: int = 50,
        n_candidates_per_query: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Generate 33-D feature matrix X, relevance labels y in {0, 1, 2, 3}, and query groups.

        Parameters
        ----------
        n_queries : int
            Number of query spectra.
        n_candidates_per_query : int
            Number of candidate structures retrieved per query.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[int]]
            - X: float32 matrix of shape (n_queries * n_candidates_per_query, 33).
            - y: int32 relevance vector of shape (n_queries * n_candidates_per_query,).
            - groups: query group sizes of length n_queries.
        """
        if n_queries <= 0 or n_candidates_per_query <= 0:
            return (
                np.empty((0, 33), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        total_rows = n_queries * n_candidates_per_query
        X = np.zeros((total_rows, 33), dtype=np.float32)
        y = np.zeros(total_rows, dtype=np.int32)
        groups: List[int] = [n_candidates_per_query] * n_queries

        for q_idx in range(n_queries):
            # 70% intra-formula constitutional isomers, 30% cross-formula mass distractors.
            # (q_idx + 7) % 10 < 7 guarantees exactly 70% intra-formula groups across every 10 queries,
            # while leaving query 0 as cross-formula for test backward compatibility.
            is_intra = ((q_idx + 7) % 10 < 7)
            start_row = q_idx * n_candidates_per_query
            true_idx = start_row
            analog_idx = start_row + 1 if n_candidates_per_query > 1 else -1
            weak_indices = (
                [start_row + c for c in range(2, min(4, n_candidates_per_query))]
                if n_candidates_per_query > 2
                else []
            )

            for c_idx in range(n_candidates_per_query):
                curr_row = start_row + c_idx
                # Base random features in [0.0, 0.35]
                feat = self.rng.uniform(0.0, 0.35, size=33).astype(np.float32)

                if curr_row == true_idx:
                    # Ground truth candidate (label = 3)
                    feat[0] = self.rng.uniform(0.85, 0.99)  # dreams_cosine
                    feat[1] = self.rng.uniform(0.80, 0.98)  # entropy_similarity
                    feat[2] = feat[0] * feat[1]            # cross-product
                    feat[3] = self.rng.uniform(0.80, 0.95)  # mist_tanimoto
                    feat[4] = self.rng.uniform(1.20, 2.00)  # np_score
                    feat[5] = feat[3] * feat[4]
                    feat[6] = 1.0                           # mass_ppm_score (0 ppm error)
                    feat[7] = 1.0                           # abs_da_score
                    feat[8] = self.rng.uniform(0.70, 0.95)  # fragment_match_ratio
                    feat[9] = float(self.rng.randint(2, 5)) # cross_track_agreement_count
                    feat[10] = 1.0 if feat[9] >= 2.0 else 0.0
                    feat[11] = 1.0                          # formula_rank = 1
                    feat[31] = self.rng.uniform(0.75, 0.98) # composite_prior_confidence (botanical prior)
                    y[curr_row] = 3
                elif curr_row == analog_idx:
                    # Near structural analog (label = 2)
                    feat[0] = self.rng.uniform(0.70, 0.85)  # dreams_cosine
                    feat[1] = self.rng.uniform(0.65, 0.80)  # entropy_similarity
                    feat[2] = feat[0] * feat[1]
                    feat[3] = self.rng.uniform(0.60, 0.75)  # mist_tanimoto
                    feat[4] = self.rng.uniform(0.80, 1.30)  # np_score
                    feat[5] = feat[3] * feat[4]
                    if is_intra:
                        feat[6] = 1.0                       # identical exact mass (0 ppm error)
                        feat[7] = 1.0
                        feat[11] = 1.0                      # identical formula rank
                    else:
                        feat[6] = 0.8                       # mass_ppm_score
                        feat[7] = 0.8
                        feat[11] = 0.5                      # formula_rank = 2
                    feat[8] = self.rng.uniform(0.50, 0.70)  # fragment_match_ratio
                    feat[9] = float(self.rng.randint(1, 3))
                    feat[10] = 1.0 if feat[9] >= 2.0 else 0.0
                    feat[31] = self.rng.uniform(0.50, 0.75) # composite_prior_confidence
                    y[curr_row] = 2
                elif curr_row in weak_indices:
                    # Weak candidate match (label = 1)
                    feat[0] = self.rng.uniform(0.50, 0.65)
                    feat[1] = self.rng.uniform(0.45, 0.60)
                    feat[2] = feat[0] * feat[1]
                    feat[3] = self.rng.uniform(0.40, 0.55)
                    feat[4] = self.rng.uniform(0.40, 0.80)
                    feat[5] = feat[3] * feat[4]
                    if is_intra:
                        feat[6] = 1.0                       # identical exact mass
                        feat[7] = 1.0
                        feat[11] = 1.0                      # identical formula rank
                    else:
                        feat[6] = 0.5                       # mass_ppm_score
                        feat[7] = 0.5
                        feat[11] = 0.33                     # formula_rank = 3
                    feat[8] = self.rng.uniform(0.25, 0.45)  # fragment_match_ratio
                    feat[9] = 1.0
                    feat[10] = 0.0
                    feat[31] = self.rng.uniform(0.25, 0.50) # composite_prior_confidence
                    y[curr_row] = 1
                else:
                    # Noise / distractor candidate (label = 0)
                    feat[2] = feat[0] * feat[1]
                    feat[5] = feat[3] * feat[4]
                    feat[8] = self.rng.uniform(0.05, 0.25)  # low fragment_match_ratio
                    feat[10] = 0.0
                    if is_intra:
                        feat[6] = 1.0                       # identical exact mass for constitutional isomer distractor
                        feat[7] = 1.0
                        feat[11] = 1.0                      # identical formula rank
                    else:
                        feat[11] = 0.1                      # formula_rank >= 10
                    feat[31] = self.rng.uniform(0.02, 0.25) # low composite_prior_confidence
                    y[curr_row] = 0

                # Clear track one-hot slots (12, 13, 14, 15, 32)
                for tidx in [12, 13, 14, 15, 32]:
                    feat[tidx] = 0.0
                # Assign one active track
                track_idx = int(self.rng.choice([12, 13, 14, 15, 32]))
                feat[track_idx] = 1.0

                X[curr_row] = feat

        assert not np.isnan(X).any(), "NaN found in generated feature matrix X"
        assert not np.isinf(X).any(), "Inf found in generated feature matrix X"
        return X, y, groups


class AuthenticRankerDatasetBuilder:
    """Extracts authentic 33-dimensional feature matrices, relevance labels, and query groups from raw training spectra and candidate database."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        meta_ranker: Optional[GBDTMetaRanker] = None,
        random_state: int = 42,
    ) -> None:
        self.db_path = db_path
        self.meta_ranker = meta_ranker or GBDTMetaRanker()
        self.router = MISTFormulaRouter()
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        self.botanical_scorer = BotanicalSubstructureScorer()

    def extract_query_candidates_and_features(
        self,
        query_row: pd.Series,
        db_searcher: SoftDatabaseSearcher,
        db_query_fp: np.ndarray,
        candidates_per_query: int = 25,
        fragment_library: Optional[Dict[int, List[str]]] = None,
        holdout_inchikey14s: Optional[Set[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and labels for a single authentic query spectrum."""
        holdouts = holdout_inchikey14s or set()
        true_smi = str(query_row.get("normalized_smiles", "")).strip()
        true_ik14 = str(query_row.get("inchikey14", "")).strip()
        true_formula = str(query_row.get("molecular_formula", "")).strip()

        if not true_smi or not true_ik14 or not true_formula or true_ik14 in holdouts:
            return np.empty((0, 33), dtype=np.float32), np.empty((0,), dtype=np.int32)

        try:
            true_mol = Chem.MolFromSmiles(true_smi)
        except Exception:
            true_mol = None

        if true_mol is None:
            return np.empty((0, 33), dtype=np.float32), np.empty((0,), dtype=np.int32)

        raw_mz = query_row.get("precursor_mz", 300.0)
        raw_adduct = query_row.get("adduct", "[M+H]+")
        try:
            neutral_mass = calculate_neutral_mass(float(raw_mz), str(raw_adduct))
        except Exception:
            neutral_mass = float(raw_mz) - 1.007825

        raw_mzs = query_row.get("ms2_mzs", [])
        raw_ints = query_row.get("ms2_normalized_intensities", [])
        mzs = np.asarray(raw_mzs, dtype=np.float64) if raw_mzs is not None else np.zeros(0, dtype=np.float64)
        ints = np.asarray(raw_ints, dtype=np.float64) if raw_ints is not None else np.zeros(0, dtype=np.float64)
        fused_peaks = (
            np.column_stack([mzs, ints])
            if len(mzs) > 0 and len(ints) == len(mzs)
            else np.zeros((0, 2), dtype=np.float64)
        )

        true_fp = self.morgan_gen.GetFingerprint(true_mol)

        # 1. Exact formula candidates (constitutional isomers sharing exact mass)
        exact_cands = db_searcher.search_formulas([true_formula], query_fp=db_query_fp)
        exact_cands = [c for c in exact_cands if c.inchikey14 == true_ik14 or c.inchikey14 not in holdouts]
        cand_iks = {c.inchikey14 for c in exact_cands}

        # Inject ground-truth molecule if not present
        gt_cand = None
        for c in exact_cands:
            if c.inchikey14 == true_ik14:
                gt_cand = c
                break
        if gt_cand is None:
            gt_popcount = float(true_fp.GetNumOnBits()) / 2048.0 if hasattr(true_fp, "GetNumOnBits") else 0.10
            gt_cand = DBCandidate(
                smiles=true_smi,
                inchikey14=true_ik14,
                tanimoto_score=gt_popcount,
                source_tier="track2_db",
            )
            exact_cands.append(gt_cand)
            cand_iks.add(true_ik14)

        # 2. Neighbor formula candidates (+/- 1H, +/- 1O) for mass-shift negative training
        neighbor_formulas = [f for f in self.router.expand_neighborhood(true_formula) if f != true_formula]
        neighbor_cands = db_searcher.search_formulas(neighbor_formulas, query_fp=db_query_fp) if neighbor_formulas else []
        neighbor_cands = [c for c in neighbor_cands if c.inchikey14 not in cand_iks and c.inchikey14 not in holdouts]
        for c in neighbor_cands:
            cand_iks.add(c.inchikey14)

        # 3. Distractor candidates from fallback scaffolds
        distractor_cands = [
            DBCandidate(smiles=s, inchikey14=ik, tanimoto_score=0.01, source_tier="track2_fallback")
            for s, ik, _ in getattr(db_searcher, "fallback_scaffolds", [])[:5]
            if ik not in cand_iks and ik not in holdouts
        ]

        # Allocate slots: ensure gt_cand + exact isomers (up to 16) + neighbor cands (up to 8) + distractors (up to 4)
        exact_isomers = [c for c in exact_cands if c.inchikey14 != true_ik14]
        if len(exact_isomers) > 16:
            perm_iso = self.rng.permutation(len(exact_isomers))
            exact_isomers = [exact_isomers[i] for i in perm_iso[:16]]
        if len(neighbor_cands) > 8:
            perm_nbr = self.rng.permutation(len(neighbor_cands))
            neighbor_cands = [neighbor_cands[i] for i in perm_nbr[:8]]
        if len(distractor_cands) > 4:
            perm_dis = self.rng.permutation(len(distractor_cands))
            distractor_cands = [distractor_cands[i] for i in perm_dis[:4]]

        selected_cands = [gt_cand] + exact_isomers + neighbor_cands + distractor_cands
        if len(selected_cands) > candidates_per_query:
            rest = selected_cands[1:]
            perm_rest = self.rng.permutation(len(rest))
            selected_cands = [gt_cand] + [rest[i] for i in perm_rest[: (candidates_per_query - 1)]]

        if not selected_cands:
            return np.empty((0, 33), dtype=np.float32), np.empty((0,), dtype=np.int32)

        # Shuffle candidates within the query so true candidate is not always at index 0
        shuffle_idx = self.rng.permutation(len(selected_cands))
        selected_cands = [selected_cands[i] for i in shuffle_idx]

        # Calculate shared neutral loss matches and shared fragment ratio from observed fused peaks (aligned with src/pipeline.py)
        shared_frag_ratio = 0.5
        shared_nl_matches = 1
        if fused_peaks is not None and len(fused_peaks) > 0 and fragment_library:
            matched_nl = 0
            for p_row in fused_peaks:
                p_mz = float(p_row[0])
                nl = neutral_mass - p_mz
                if nl > 0.5:
                    nl_key = round(nl * 10000)
                    if nl_key in fragment_library:
                        matched_nl += 1
            shared_frag_ratio = min(1.0, float(matched_nl) / max(1, len(fused_peaks)))
            shared_nl_matches = matched_nl

        n_cands = len(selected_cands)
        X_query = np.zeros((n_cands, 33), dtype=np.float32)
        y_query = np.zeros(n_cands, dtype=np.int32)

        for idx, cand in enumerate(selected_cands):
            c_smi = cand.smiles
            c_ik = cand.inchikey14

            c_mol = None
            if c_smi:
                try:
                    c_mol = Chem.MolFromSmiles(c_smi)
                except Exception:
                    c_mol = None

            cand_ppm = 2.0
            cand_abs_da = 0.001
            np_score = 1.0
            cand_frag_ratio = shared_frag_ratio
            bot_score = 0.5
            mw_norm = min(1.0, neutral_mass / 1000.0)
            hbd = 2.0
            hba = 4.0
            rot_b = 3.0
            tpsa = 0.35
            arom_ratio = 0.5

            if c_mol is not None:
                try:
                    cand_mw = float(ExactMolWt(c_mol))
                    cand_abs_da = abs(cand_mw - neutral_mass)
                    cand_ppm = (cand_abs_da / max(neutral_mass, 1.0)) * 1e6
                    n_c = max(1, c_smi.count("C") + c_smi.count("c"))
                    n_o = c_smi.count("O") + c_smi.count("o")
                    n_n = c_smi.count("N") + c_smi.count("n")
                    np_score = float((n_o * 1.5 + n_n * 1.2) / n_c)
                    cand_in_silico = _compute_candidate_fragment_match_ratio(
                        mol=c_mol, fused_peaks=fused_peaks, neutral_mass=neutral_mass
                    )
                    cand_frag_ratio = min(0.95, max(shared_frag_ratio, shared_frag_ratio + cand_in_silico * 0.45))
                    mw_norm = min(1.0, cand_mw / 1000.0)
                    hbd = float(Lipinski.NumHDonors(c_mol))
                    hba = float(Lipinski.NumHAcceptors(c_mol))
                    rot_b = float(Lipinski.NumRotatableBonds(c_mol))
                    tpsa = float(Descriptors.TPSA(c_mol) / 200.0)
                    arom_ratio = float(Descriptors.FractionCSP3(c_mol))
                except Exception:
                    pass

            if c_mol is not None and hasattr(self, "botanical_scorer"):
                try:
                    bot_score = self.botanical_scorer.score_candidate(c_mol, fused_peaks, neutral_mass)
                except Exception:
                    bot_score = 0.5
            elif c_smi and hasattr(self, "botanical_scorer"):
                try:
                    bot_score = self.botanical_scorer.score_candidate(c_smi, fused_peaks, neutral_mass)
                except Exception:
                    bot_score = 0.5

            # Downweight synthetic distractors (aligned with src/pipeline.py)
            if bot_score <= 0.15:
                np_score = min(np_score, 0.1)

            # Blend bot_score into cand_frag_ratio and mist_tanimoto (aligned with src/pipeline.py)
            cand_frag_ratio = float(np.clip(0.40 * cand_frag_ratio + 0.60 * bot_score, 0.001, 0.999))
            base_mist = cand_frag_ratio if "db" in cand.source_tier else getattr(cand, "tanimoto_score", 0.5)
            blended_mist = float(np.clip(0.40 * base_mist + 0.60 * bot_score, 0.001, 0.999))

            feat = self.meta_ranker.extract_feature_vector(
                dreams_cosine=cand.tanimoto_score if "dreams" in cand.source_tier else 0.5,
                mist_tanimoto=blended_mist,
                ppm_error=cand_ppm,
                abs_da_error=cand_abs_da,
                fragment_match_ratio=cand_frag_ratio,
                np_score=np_score,
                entropy_similarity=0.8,
                cross_track_agreement_count=1,
                formula_rank=1,
                source_track=cand.source_tier,
                molecular_weight_norm=mw_norm,
                h_bond_donors_est=hbd,
                h_bond_acceptors_est=hba,
                rotatable_bonds_est=rot_b,
                tpsa_est_norm=tpsa,
                aromatic_ring_ratio=arom_ratio,
                neutral_loss_match_count=shared_nl_matches,
                composite_prior_confidence=bot_score,
            )
            X_query[idx] = feat

            # Relevance assignment
            if c_ik == true_ik14:
                y_query[idx] = 3
            elif c_mol is not None:
                try:
                    c_fp = self.morgan_gen.GetFingerprint(c_mol)
                    sim = float(DataStructs.TanimotoSimilarity(true_fp, c_fp))
                    if sim >= 0.65:
                        y_query[idx] = 2
                    elif sim >= 0.40:
                        y_query[idx] = 1
                    else:
                        y_query[idx] = 0
                except Exception:
                    y_query[idx] = 0
            else:
                y_query[idx] = 0

        return X_query, y_query

    def build_dataset_from_dataframe(
        self,
        df: pd.DataFrame,
        db_searcher: SoftDatabaseSearcher,
        n_queries: int = 500,
        candidates_per_query: int = 25,
        holdout_inchikey14s: Optional[Set[str]] = None,
        fragment_library: Optional[Dict[int, List[str]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Build multimodal ranking dataset from an authentic training DataFrame."""
        if getattr(db_searcher, "repo", None) is not None:
            db_query_fp = np.full(32, np.iinfo(np.uint64).max, dtype=np.uint64)
        else:
            db_query_fp = np.ones(3, dtype=np.float32)

        holdouts = holdout_inchikey14s or set()
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        groups: List[int] = []

        seen_queries = set()

        for idx in range(len(df)):
            if len(groups) >= n_queries:
                break

            row = df.iloc[idx]
            ik14 = str(row.get("inchikey14", "")).strip()
            if not ik14 or ik14 in holdouts or ik14 in seen_queries:
                continue

            X_q, y_q = self.extract_query_candidates_and_features(
                query_row=row,
                db_searcher=db_searcher,
                db_query_fp=db_query_fp,
                candidates_per_query=candidates_per_query,
                fragment_library=fragment_library,
                holdout_inchikey14s=holdouts,
            )

            # Require at least 2 candidates and at least one positive label for learning-to-rank
            if len(y_q) >= 2 and (y_q > 0).any():
                all_X.append(X_q)
                all_y.append(y_q)
                groups.append(len(y_q))
                seen_queries.add(ik14)

        if not all_X:
            return (
                np.empty((0, 33), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        assert not np.isnan(X).any(), "NaN found in authentic feature matrix X"
        assert not np.isinf(X).any(), "Inf found in authentic feature matrix X"
        return X, y, groups
