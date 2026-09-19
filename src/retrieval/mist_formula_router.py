import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np


@dataclass
class FormulaRoutingDecision:
    formulas: List[str]
    entropy: float
    use_formula_free_fallback: bool


class MISTFormulaRouter:
    """Soft formula posterior expansion and entropy-triggered fallback router."""

    def __init__(self, entropy_threshold: float = 1.2):
        self.entropy_threshold = entropy_threshold

    def calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy H = - sum(p_i * ln(p_i)) over non-zero probabilities."""
        if not probabilities:
            return 0.0
        p = np.array(probabilities, dtype=np.float64)
        p = p[p > 0]
        if len(p) <= 1:
            return 0.0
        total = np.sum(p)
        if total <= 0:
            return 0.0
        p = p / total
        return float(-np.sum(p * np.log(p + 1e-12)))

    def expand_neighborhood(self, formula: str) -> Set[str]:
        """Expand formula by +/- 1H and +/- 1O neighbors using Hill system ordering."""
        if not formula or not isinstance(formula, str):
            return set()

        elements: Dict[str, int] = {}
        for elem, cnt in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
            if elem:
                count = int(cnt) if cnt else 1
                elements[elem] = elements.get(elem, 0) + count

        if not elements:
            return set()

        def format_form(c_dict: Dict[str, int]) -> str:
            # Standard Hill system order: C, then H, then remaining elements in alphabetical order
            if 'C' in c_dict and c_dict['C'] > 0:
                order = ['C', 'H'] + sorted([k for k in c_dict if k not in ['C', 'H']])
            else:
                order = sorted(list(c_dict.keys()))
            parts = []
            for k in order:
                if k in c_dict and c_dict[k] > 0:
                    cnt = c_dict[k]
                    parts.append(f"{k}{cnt if cnt > 1 else ''}")
            return "".join(parts)

        expanded: Set[str] = set()
        base_formatted = format_form(elements)
        if base_formatted:
            expanded.add(base_formatted)
        expanded.add(formula)

        # +/- 1H
        for dh in [-1, 1]:
            cand = elements.copy()
            cand['H'] = max(0, cand.get('H', 0) + dh)
            form_str = format_form(cand)
            if form_str:
                expanded.add(form_str)

        # +/- 1O
        for do in [-1, 1]:
            cand = elements.copy()
            cand['O'] = max(0, cand.get('O', 0) + do)
            form_str = format_form(cand)
            if form_str:
                expanded.add(form_str)

        return expanded

    def route_formula_distribution(
        self, top_candidates: List[Tuple[str, float]]
    ) -> FormulaRoutingDecision:
        """Determine formula candidate set and fallback based on prediction entropy."""
        if not top_candidates:
            return FormulaRoutingDecision(
                formulas=[],
                entropy=0.0,
                use_formula_free_fallback=True,
            )

        probs = [prob for _, prob in top_candidates]
        entropy = self.calculate_entropy(probs)
        use_fallback = entropy > self.entropy_threshold

        expanded_set: Set[str] = set()
        for form, _ in top_candidates[:3]:
            expanded_set.update(self.expand_neighborhood(form))

        return FormulaRoutingDecision(
            formulas=sorted(list(expanded_set)),
            entropy=entropy,
            use_formula_free_fallback=use_fallback,
        )
