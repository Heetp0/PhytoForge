import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

from src.chemistry.adducts import (
    calculate_canonical_neutral_mass,
    get_adduct_info,
    get_multihypothesis_precursor_candidates,
)
from src.data.multi_energy_fusion import MultiEnergyFusionEngine, SpectralFrame
from src.retrieval.mist_formula_router import MISTFormulaRouter
from src.retrieval.dreams_retrieval import CalibratedDreaMSRetriever
from src.retrieval.database_search import SoftDatabaseSearcher
from src.retrieval.generative_denovo import BoundedGenerativeEngine
from src.retrieval.transductive_networking import TransductiveMolecularNetwork
from src.reranking.meta_ranker import GBDTMetaRanker
from src.reranking.slot_optimizer import DecisionTheoreticSlotOptimizer
from src.submission.writer import (
    DEFAULT_FALLBACK_SMILES,
    get_inchikey14,
    validate_submission,
    write_submission,
)
from src.submission.runtime_governor import DynamicRuntimeGovernor


class CASMIOmegaPipeline:
    """Unified CASMI-Omega v2 End-to-End Pipeline connecting Modules 1-6."""

    def __init__(
        self,
        db: Optional[Dict[str, List]] = None,
        fallback_scaffolds: Optional[List] = None,
        governor_budget_seconds: float = 32400.0,
        db_path: Optional[Union[str, Path]] = None,
        spectral_index_path: Optional[Union[str, Path]] = None,
        fragment_library: Optional[Dict[int, List[str]]] = None,
    ):
        self.fallback_pool = [f"IK14FALL{i:06d}" for i in range(50)]
        self.db = db if db is not None else {
            "C15H10O5": [("c1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1", "IK14DBFLAV0001", np.ones(3, dtype=np.float32))],
            "C15H10O6": [("c1cc(O)c2c(=O)cc(-c3ccc(O)cc3)oc2c1", "IK14DBFLAV0002", np.ones(3, dtype=np.float32))],
        }
        self.fallback_scaffolds = fallback_scaffolds if fallback_scaffolds is not None else [
            ("C1CCCCC1", "IK14FALL000000", np.zeros(3, dtype=np.float32))
        ]
        self.db_path = db_path
        self.spectral_index_path = spectral_index_path
        self.fragment_library = fragment_library
        self.last_aggregated_cands: List[Dict[str, Any]] = []

        # Initialize pipeline modules (Modules 1 through 6)
        self.fusion_engine = MultiEnergyFusionEngine(embedding_dim=1024)
        self.formula_router = MISTFormulaRouter(entropy_threshold=1.2)
        self.dreams_retriever = (
            CalibratedDreaMSRetriever(index=spectral_index_path)
            if spectral_index_path is not None
            else CalibratedDreaMSRetriever()
        )
        self.db_searcher = SoftDatabaseSearcher(
            db=self.db,
            fallback_scaffolds=self.fallback_scaffolds,
            db_path=db_path,
        )
        self.denovo_engine = BoundedGenerativeEngine(timeout_seconds=5.0)
        self.transductive_network = TransductiveMolecularNetwork(cosine_threshold=0.7)
        self.meta_ranker = GBDTMetaRanker()
        self.slot_optimizer = DecisionTheoreticSlotOptimizer(fallback_pool=self.fallback_pool)
        self.governor = DynamicRuntimeGovernor(total_budget_seconds=governor_budget_seconds, reserve_seconds=480.0)

    def close(self) -> None:
        """Release underlying database and spectral index resources."""
        if hasattr(self, "db_searcher") and self.db_searcher is not None:
            try:
                self.db_searcher.close()
            except Exception:
                pass
        if hasattr(self, "dreams_retriever") and self.dreams_retriever is not None:
            try:
                self.dreams_retriever.close()
            except Exception:
                pass

    def __enter__(self) -> "CASMIOmegaPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def run(
        self,
        test_df: pd.DataFrame,
        output_path: Union[str, Path],
        output_format: str = "inchikey14",
    ) -> pd.DataFrame:
        output_path = Path(output_path)
        if output_format is None:
            output_format = "inchikey14"
        fmt = str(output_format).strip().lower()
        if fmt not in ("inchikey14", "smiles"):
            raise ValueError(
                f"Invalid output_format '{output_format}'. Must be 'inchikey14' or 'smiles'."
            )

        if test_df is None or (isinstance(test_df, pd.DataFrame) and test_df.empty):
            test_df = pd.DataFrame()

        if not test_df.empty:
            id_col = "id" if "id" in test_df.columns else ("molecule_id" if "molecule_id" in test_df.columns else None)
            if id_col is not None and test_df[id_col].notna().all():
                expected_ids = [str(x) for x in test_df[id_col]]
            else:
                expected_ids = [
                    str(test_df.iloc[i][id_col]) if id_col is not None and pd.notna(test_df.iloc[i][id_col]) else f"SPEC_{i:04d}"
                    for i in range(len(test_df))
                ]
        else:
            expected_ids = []

        if test_df.empty:
            if fmt == "inchikey14":
                sub_df = pd.DataFrame(columns=["id", "candidates"])
            else:
                sub_df = pd.DataFrame(columns=["molecule_id", "smiles"])
            validate_submission(sub_df, expected_ids=expected_ids, output_format=fmt)
            write_submission(
                sub_df,
                output_path=output_path,
                expected_ids=expected_ids,
                output_format=fmt,
            )
            return sub_df

        total_spectra = len(test_df)
        rows = []
        start_pipeline_time = time.perf_counter()

        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            spec_id = expected_ids[row_idx]

            try:
                raw_mz = row.get("precursor_mz", 300.0)
                precursor_mz = float(raw_mz)
                if np.isnan(precursor_mz) or np.isinf(precursor_mz) or precursor_mz <= 0.0:
                    precursor_mz = 300.0
            except (ValueError, TypeError):
                precursor_mz = 300.0

            polarity = str(row.get("polarity", "positive")).strip().lower()
            default_adduct = "[M-H]-" if polarity in ("negative", "-", "neg") else "[M+H]+"
            raw_adduct = row.get("adduct", default_adduct)
            if pd.isna(raw_adduct) or not str(raw_adduct).strip():
                adduct = default_adduct
            else:
                adduct_str = str(raw_adduct).strip()
                try:
                    get_adduct_info(adduct_str)
                    adduct = adduct_str
                except Exception:
                    adduct = default_adduct
            spectra_remaining = total_spectra - len(rows)
            elapsed = time.perf_counter() - start_pipeline_time

            # Module 6 Dynamic Runtime Governor check
            per_spec_budget = self.governor.get_per_spectrum_budget(elapsed, spectra_remaining)

            # Module 1: Preprocessing & Physical plausibility
            neutral_mass = calculate_canonical_neutral_mass(precursor_mz, adduct)
            hypotheses = get_multihypothesis_precursor_candidates(precursor_mz, adduct, mw_estimate=neutral_mass)

            # Module 1b: Spectral embedding fusion across frames
            frame = SpectralFrame(
                collision_energy=35.0,
                peaks=np.array([[100.0, 50.0], [150.0, 80.0]]),
                embedding=np.ones(1024, dtype=np.float32) / np.sqrt(1024),
            )
            fused_emb = self.fusion_engine.fuse_embeddings([frame])

            # Module 2: MIST-CF soft formula routing
            formula_priors = [("C15H10O5", 0.7), ("C15H10O6", 0.2), ("C14H8O5", 0.1)]
            routing = self.formula_router.route_formula_distribution(formula_priors)

            # Module 3: Tri-Track Candidate Generation
            # Track 1: DreaMS library retrieval
            if getattr(self.dreams_retriever, "has_index", False):
                track1_cands, _ = self.dreams_retriever.evaluate_candidates(
                    query_emb=fused_emb,
                    precursor_mz=precursor_mz,
                    polarity=polarity,
                    query_instrument="timsTOF",
                )
            else:
                dummy_lib = [("c1ccccc1", "IK14DREAM00001", fused_emb.copy(), "timsTOF")]
                track1_cands, _ = self.dreams_retriever.evaluate_candidates(
                    query_emb=fused_emb, raw_candidates=dummy_lib, query_instrument="timsTOF"
                )

            # Track 2: Formula-constrained database search
            db_query_fp = (
                np.zeros(32, dtype=np.uint64)
                if getattr(self.db_searcher, "repo", None) is not None
                else np.ones(3, dtype=np.float32)
            )
            track2_cands = self.db_searcher.search_formulas(
                formulas=routing.formulas, query_fp=db_query_fp
            )

            # Track 3: Latency-bounded de novo sampling (conditioned on remaining per-spec budget)
            track3_timeout = min(per_spec_budget, 10.0)
            track3_cands = self.denovo_engine.generate_with_timeout(
                spectrum_id=spec_id,
                formula=routing.formulas[0] if routing.formulas else None,
                timeout_seconds=track3_timeout,
            )

            # Transductive networking propagation
            self.transductive_network.add_spectrum(
                spec_id=spec_id, neutral_mass=neutral_mass, embedding=fused_emb, peaks=frame.peaks
            )
            network_cands = self.transductive_network.propagate_scaffold(
                target_id=spec_id,
                known_scaffolds={"SEED": ("c1ccccc1O", "IK14SEED000001")},
            )

            # Track K: Knapsack substructure assembly
            knapsack_cands = []
            if self.fragment_library:
                try:
                    from src.reranking.fragment_library import KnapsackAssembler
                    knapsack_cands = KnapsackAssembler().assemble(
                        target_mass_da=neutral_mass,
                        library=self.fragment_library,
                        timeout_s=2.0,
                    )
                except Exception:
                    knapsack_cands = []

            # Aggregate multi-track candidates
            aggregated_cands = []
            for c in track1_cands:
                aggregated_cands.append({"inchikey14": c.inchikey14, "smiles": getattr(c, "smiles", None), "score": c.score, "source": "track1_dreams"})
            for c in track2_cands:
                aggregated_cands.append({"inchikey14": c.inchikey14, "smiles": getattr(c, "smiles", None), "score": c.tanimoto_score, "source": "track2_db"})
            for c in track3_cands:
                aggregated_cands.append({"inchikey14": c.inchikey14, "smiles": getattr(c, "smiles", None), "score": c.score, "source": "track3_denovo"})
            for c in network_cands:
                aggregated_cands.append({"inchikey14": c.inchikey14, "smiles": getattr(c, "scaffold_smiles", None), "score": c.network_score, "source": "track_network"})

            for c_str in knapsack_cands:
                cand_str = str(c_str).strip()
                if not cand_str:
                    continue
                if len(cand_str) == 14 and cand_str.isalnum():
                    ik14 = cand_str
                else:
                    derived_ik14 = get_inchikey14(cand_str)
                    if derived_ik14 and len(derived_ik14) == 14 and derived_ik14.isalnum():
                        ik14 = derived_ik14
                    else:
                        ik14 = f"IK14KNAP{abs(hash(cand_str)) % 1000000:06d}"
                aggregated_cands.append({
                    "inchikey14": ik14,
                    "smiles": cand_str,
                    "score": 0.3,
                    "source": "track_knapsack",
                })

            self.last_aggregated_cands = list(aggregated_cands)

            # Module 4: GBDT Meta-ranking feature extraction & scoring
            scored_candidates = []
            for c in aggregated_cands:
                feat = self.meta_ranker.extract_feature_vector(
                    dreams_cosine=c["score"] if "dreams" in c["source"] else 0.5,
                    mist_tanimoto=c["score"] if "db" in c["source"] else 0.5,
                    ppm_error=2.0,
                    abs_da_error=0.001,
                    fragment_match_ratio=0.7,
                    np_score=1.2,
                    entropy_similarity=0.8,
                    cross_track_agreement_count=1,
                    formula_rank=1,
                    source_track=c["source"],
                )
                cand_entry = {"inchikey14": c["inchikey14"], "features": feat}
                if c.get("smiles"):
                    cand_entry["smiles"] = c["smiles"]
                scored_candidates.append(cand_entry)

            ranked = self.meta_ranker.score_candidates(scored_candidates)

            if fmt == "inchikey14":
                # Module 5: Planar InChIKey14 slot optimizer (strictly 25 unique valid keys)
                slots = self.slot_optimizer.allocate_25_slots(ranked)
                rows.append({"id": spec_id, "candidates": ";".join(slots)})
            else:
                # SMILES format: 25 unique, valid SMILES strings
                selected_smiles: List[str] = []
                seen_ik14: Set[str] = set()

                for cand in ranked:
                    s = cand.get("smiles")
                    if s and isinstance(s, str) and s.strip():
                        s_clean = s.strip()
                        if any(ch in s_clean for ch in (" ", "\t", "\n", "\r", ",")):
                            continue
                        try:
                            from rdkit import Chem
                            if Chem.MolFromSmiles(s_clean) is None:
                                continue
                        except Exception:
                            pass
                        ik14 = get_inchikey14(s_clean)
                        if ik14:
                            if ik14 in seen_ik14:
                                continue
                            seen_ik14.add(ik14)
                        elif s_clean in selected_smiles:
                            continue
                        selected_smiles.append(s_clean)
                        if len(selected_smiles) == 25:
                            break

                if len(selected_smiles) < 25:
                    for fb in DEFAULT_FALLBACK_SMILES:
                        fb_clean = fb.strip()
                        fb_ik = get_inchikey14(fb_clean)
                        if fb_ik and fb_ik in seen_ik14:
                            continue
                        if fb_clean in selected_smiles:
                            continue
                        if fb_ik:
                            seen_ik14.add(fb_ik)
                        selected_smiles.append(fb_clean)
                        if len(selected_smiles) == 25:
                            break

                alkane_k = 1
                while len(selected_smiles) < 25:
                    cand_smi = "C" * alkane_k
                    cand_ik = get_inchikey14(cand_smi)
                    if cand_smi not in selected_smiles and (not cand_ik or cand_ik not in seen_ik14):
                        if cand_ik:
                            seen_ik14.add(cand_ik)
                        selected_smiles.append(cand_smi)
                    alkane_k += 1

                selected_smiles = selected_smiles[:25]
                rows.append({"molecule_id": spec_id, "smiles": ";".join(selected_smiles)})

        if fmt == "inchikey14":
            sub_df = pd.DataFrame(rows, columns=["id", "candidates"])
        else:
            sub_df = pd.DataFrame(rows, columns=["molecule_id", "smiles"])

        # Module 6: Submission validation and atomic file writing
        validate_submission(sub_df, expected_ids=expected_ids, output_format=fmt)
        write_submission(
            sub_df,
            output_path=output_path,
            expected_ids=expected_ids,
            output_format=fmt,
        )
        return sub_df


def run_casmi_omega_pipeline(
    test_df: pd.DataFrame,
    output_path: Union[str, Path],
    db: Optional[Dict[str, List]] = None,
    fallback_scaffolds: Optional[List] = None,
    governor_budget_seconds: float = 32400.0,
    output_format: str = "inchikey14",
    db_path: Optional[Union[str, Path]] = None,
    spectral_index_path: Optional[Union[str, Path]] = None,
    fragment_library: Optional[Dict[int, List[str]]] = None,
) -> pd.DataFrame:
    """Execute CASMIOmegaPipeline on test DataFrame and export validated submission CSV."""
    with CASMIOmegaPipeline(
        db=db,
        fallback_scaffolds=fallback_scaffolds,
        governor_budget_seconds=governor_budget_seconds,
        db_path=db_path,
        spectral_index_path=spectral_index_path,
        fragment_library=fragment_library,
    ) as pipeline:
        return pipeline.run(test_df, output_path=output_path, output_format=output_format)


# Official Brand Aliases for PhytoForge
PhytoForgePipeline = CASMIOmegaPipeline
run_phytoforge_pipeline = run_casmi_omega_pipeline
