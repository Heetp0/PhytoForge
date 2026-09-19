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

    def allocate_25_slots(self, scored_candidates: List[Dict[str, Any]]) -> List[str]:
        """Allocate exactly 25 unique, strictly 14-char alphanumeric InChIKey14 slots.

        Strategy:
        - Prioritizes candidates by score descending if scores are present.
        - Slots 1-3 (Exploitation): Top calibrated candidate predictions.
        - Slots 4-25: Remaining unique candidates.
        - Fallback: COCONUT NP diversity bank when candidate pool is depleted.
        - Emergency padding: Deterministic PAD00000000000 format if fallback is insufficient.

        Args:
            scored_candidates: List of candidate dicts or objects containing 'inchikey14' and optional 'score'.

        Returns:
            List of exactly 25 unique InChIKey14 strings.
        """
        slots: List[str] = []
        seen: Set[str] = set()

        def _extract_score(cand: Any) -> float:
            if isinstance(cand, dict):
                score = cand.get("score")
                if score is not None:
                    try:
                        return float(score)
                    except (ValueError, TypeError):
                        pass
            return 0.0

        def _extract_ik14(cand: Any) -> Optional[str]:
            if isinstance(cand, dict):
                val = cand.get("inchikey14")
                if isinstance(val, str):
                    return val
            elif isinstance(cand, str):
                return cand
            return None

        # Sort candidates by score descending if available, preserving stable order
        sorted_candidates = sorted(
            scored_candidates or [],
            key=_extract_score,
            reverse=True,
        )

        # Slots 1-3 (Exploitation) & remaining candidates
        for cand in sorted_candidates:
            ik14 = _extract_ik14(cand)
            if ik14 and len(ik14) == 14 and ik14.isalnum() and ik14 not in seen:
                slots.append(ik14)
                seen.add(ik14)
                if len(slots) == 25:
                    return slots

        # Slots 4-25: If candidate pool depleted, pad with fallback pool
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
