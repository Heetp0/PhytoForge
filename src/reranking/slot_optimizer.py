"""Module 5: Decision-Theoretic Planar InChIKey14 Slot Optimizer.

Allocates exactly 25 unique planar InChIKey14 prediction slots
combining top calibrated candidate predictions, natural product diversity fallback,
and deterministic emergency padding.
"""

from typing import Any, Dict, List, Optional, Set


class DecisionTheoreticSlotOptimizer:
    """Module 5: Decision-theoretic 25-slot portfolio optimizer on planar InChIKey14."""

    def __init__(self, fallback_pool: Optional[List[str]] = None) -> None:
        """Initialize slot optimizer with sanitized fallback pool.

        Args:
            fallback_pool: Optional list of fallback InChIKey14 strings (e.g. COCONUT NP diversity bank).
        """
        seen_fallback: Set[str] = set()
        cleaned_fallback: List[str] = []
        for k in fallback_pool or []:
            if isinstance(k, str) and len(k) == 14 and k.isalnum() and k not in seen_fallback:
                cleaned_fallback.append(k)
                seen_fallback.add(k)
        self.fallback_pool = cleaned_fallback

    def allocate_25_slots(
        self,
        scored_candidates: List[Dict[str, Any]],
        diversity_lambda: float = 0.88,
    ) -> List[str]:
        """Allocate exactly 25 unique, strictly 14-char alphanumeric InChIKey14 slots.

        Strategy:
        - Prioritizes candidates by score descending if scores are present.
        - Slots 1-3 (Exploitation): Top calibrated candidate predictions.
        - Slots 4-25 (Exploration & Diversity): Maximal Marginal Relevance (MMR)
          penalizing redundant identical scaffolds using Tanimoto distance.
        - Fallback: COCONUT NP diversity bank when candidate pool is depleted.
        - Emergency padding: Deterministic PAD00000000000 format if fallback is insufficient.

        Args:
            scored_candidates: List of candidate dicts or objects containing 'inchikey14' and optional 'score'.
            diversity_lambda: Trade-off parameter between relevance score and structural diversity (default 0.88).

        Returns:
            List of exactly 25 unique InChIKey14 strings.
        """
        slots: List[str] = []
        seen: Set[str] = set()

        def _extract_score(cand: Any) -> float:
            if isinstance(cand, dict):
                score = cand.get("score")
                if score is None:
                    score = cand.get("meta_score")
                if score is not None:
                    try:
                        return float(score)
                    except (ValueError, TypeError):
                        pass
            elif hasattr(cand, "score"):
                try:
                    return float(cand.score)
                except (ValueError, TypeError):
                    pass
            elif hasattr(cand, "meta_score"):
                try:
                    return float(cand.meta_score)
                except (ValueError, TypeError):
                    pass
            return 0.0

        def _extract_ik14(cand: Any) -> Optional[str]:
            if isinstance(cand, dict):
                val = cand.get("inchikey14")
                if isinstance(val, str):
                    return val
            elif hasattr(cand, "inchikey14"):
                val = getattr(cand, "inchikey14")
                if isinstance(val, str):
                    return val
            elif isinstance(cand, str):
                return cand
            return None

        def _extract_smiles(cand: Any) -> Optional[str]:
            if isinstance(cand, dict):
                val = cand.get("smiles")
                if isinstance(val, str):
                    return val
            elif hasattr(cand, "smiles"):
                val = getattr(cand, "smiles")
                if isinstance(val, str):
                    return val
            return None

        # Prepare deduplicated candidate pool
        pool: List[Dict[str, Any]] = []
        pool_iks: Set[str] = set()
        for cand in scored_candidates or []:
            ik14 = _extract_ik14(cand)
            if ik14 and len(ik14) == 14 and ik14.isalnum() and ik14 not in pool_iks:
                smi = _extract_smiles(cand)
                fp = None
                if smi and isinstance(smi, str):
                    try:
                        from rdkit import Chem
                        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
                        m = Chem.MolFromSmiles(smi.strip())
                        if m is not None:
                            gen = GetMorganGenerator(radius=2, fpSize=1024)
                            fp = gen.GetFingerprint(m)
                    except Exception:
                        fp = None
                pool.append({
                    "ik14": ik14,
                    "score": _extract_score(cand),
                    "smiles": smi,
                    "fp": fp,
                })
                pool_iks.add(ik14)

        # Sort pool by score descending
        pool.sort(key=lambda c: c["score"], reverse=True)

        selected: List[Dict[str, Any]] = []

        # Slots 1-3: Pure exploitation of top candidates
        while pool and len(slots) < min(3, 25):
            top_cand = pool.pop(0)
            slots.append(top_cand["ik14"])
            seen.add(top_cand["ik14"])
            selected.append(top_cand)

        def _normalize_score(val: float) -> float:
            if 0.0 <= val <= 1.0:
                return val
            # Stable sigmoid mapping for unbounded GBDT margins
            import math
            try:
                return 1.0 / (1.0 + math.exp(-val))
            except OverflowError:
                return 1.0 if val > 0 else 0.0

        # Slots 4-25: Maximal Marginal Relevance (MMR)
        if pool and len(slots) < 25:
            has_fps = any(c["fp"] is not None for c in pool) and any(s["fp"] is not None for s in selected)

            while pool and len(slots) < 25:
                if diversity_lambda >= 1.0 or not has_fps:
                    # Greedy by score
                    best_cand = pool.pop(0)
                else:
                    try:
                        from rdkit import DataStructs
                        best_idx = 0
                        best_mmr = -float("inf")
                        for idx, cand in enumerate(pool):
                            norm_score = _normalize_score(cand["score"])
                            if cand["fp"] is not None:
                                max_sim = max([
                                    DataStructs.TanimotoSimilarity(cand["fp"], s["fp"])
                                    for s in selected if s["fp"] is not None
                                ] or [0.0])
                            else:
                                max_sim = 0.0
                            mmr = diversity_lambda * norm_score - (1.0 - diversity_lambda) * max_sim
                            if mmr > best_mmr:
                                best_mmr = mmr
                                best_idx = idx
                        best_cand = pool.pop(best_idx)
                    except Exception:
                        best_cand = pool.pop(0)

                slots.append(best_cand["ik14"])
                seen.add(best_cand["ik14"])
                selected.append(best_cand)

        # Fallback pool padding if candidate pool depleted
        if len(slots) < 25:
            for fallback_ik14 in self.fallback_pool:
                if fallback_ik14 not in seen and len(fallback_ik14) == 14 and fallback_ik14.isalnum():
                    slots.append(fallback_ik14)
                    seen.add(fallback_ik14)
                    if len(slots) == 25:
                        return slots

        # Emergency pad if fallback pool insufficient (strictly 14-char alphanumeric)
        pad_idx = 0
        while len(slots) < 25:
            dummy = f"PAD{pad_idx:011d}"
            if dummy not in seen:
                slots.append(dummy)
                seen.add(dummy)
            pad_idx += 1

        return slots

    def optimize_slots(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """Alias for allocate_25_slots."""
        return self.allocate_25_slots(candidates)
