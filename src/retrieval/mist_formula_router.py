import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
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


class ExactMassFormulaIndex:
    """Pre-computed exact monoisotopic mass index over chemical formulas for microsecond precursor routing."""

    ELEM_WEIGHTS: Dict[str, float] = {
        "C": 12.0,
        "H": 1.00782503223,
        "O": 15.99491461957,
        "N": 14.00307400443,
        "S": 31.9720711744,
        "P": 30.9737619984,
        "F": 18.998403163,
        "Cl": 34.9688527,
        "Br": 78.9183376,
        "I": 126.904473,
    }
    PATTERN = re.compile(r"([A-Z][a-z]*)(\d*)")
    CARBON_PATTERN = re.compile(r"C(\d*)")

    def __init__(
        self,
        formulas: Optional[List[str]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.formulas: List[str] = []
        self.masses: np.ndarray = np.empty(0, dtype=np.float64)
        self.carbon_counts: np.ndarray = np.empty(0, dtype=np.int32)
        self.has_cl: np.ndarray = np.empty(0, dtype=bool)
        self.has_br: np.ndarray = np.empty(0, dtype=bool)
        self.has_f: np.ndarray = np.empty(0, dtype=bool)

        if formulas is not None:
            self._build_index(formulas)
        elif db_path is not None:
            import sqlite3
            p = Path(db_path).resolve()
            if p.exists():
                if "/kaggle/input" in str(p):
                    import shutil
                    writable_dir = Path("/tmp/phytoforge_db")
                    writable_dir.mkdir(parents=True, exist_ok=True)
                    target_copy = writable_dir / p.name
                    if not target_copy.exists():
                        shutil.copy2(p, target_copy)
                    p = target_copy
                try:
                    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
                except Exception:
                    conn = sqlite3.connect(str(p))
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT DISTINCT formula FROM candidates;")
                    forms = [r[0] for r in cur.fetchall()]
                    conn.close()
                    self._build_index(forms)
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass

    @classmethod
    def compute_formula_mass(cls, formula: str) -> float:
        """Compute exact monoisotopic neutral mass of Hill chemical formula."""
        m = 0.0
        for el, cnt in cls.PATTERN.findall(formula):
            w = cls.ELEM_WEIGHTS.get(el, 0.0)
            c = int(cnt) if cnt else 1
            m += w * c
        return m

    def _build_index(self, formulas: List[str]) -> None:
        clean_forms = [f.strip() for f in formulas if f and isinstance(f, str) and f.strip()]
        self.formulas = clean_forms
        n = len(clean_forms)
        masses = np.zeros(n, dtype=np.float64)
        carbons = np.zeros(n, dtype=np.int32)
        has_cl = np.zeros(n, dtype=bool)
        has_br = np.zeros(n, dtype=bool)
        has_f = np.zeros(n, dtype=bool)

        for i, f in enumerate(clean_forms):
            masses[i] = self.compute_formula_mass(f)
            c_matches = self.CARBON_PATTERN.findall(f)
            carbons[i] = sum(int(x or 1) for x in c_matches) if c_matches else 0
            elems = {el for el, _ in self.PATTERN.findall(f)}
            has_cl[i] = "Cl" in elems
            has_br[i] = "Br" in elems
            has_f[i] = "F" in elems

        self.masses = masses
        self.carbon_counts = carbons
        self.has_cl = has_cl
        self.has_br = has_br
        self.has_f = has_f

    def find_formulas_by_mass(
        self,
        target_mass: float,
        tolerance_ppm: float = 20.0,
        target_carbon_estimate: Optional[float] = None,
        max_results: int = 8,
        has_chlorine: bool = False,
        has_bromine: bool = False,
    ) -> List[Tuple[str, float]]:
        """Search formulas matching target monoisotopic mass within tolerance_ppm.
        
        Optionally uses target_carbon_estimate and isotope-based halogen detection to prioritize formulas.
        Returns List of (formula_str, confidence_prior) tuples.
        """
        if len(self.masses) == 0 or target_mass <= 0:
            return []

        tol_da = target_mass * (tolerance_ppm * 1e-6)
        diffs = np.abs(self.masses - target_mass)
        matching_indices = np.where(diffs <= tol_da)[0]

        if len(matching_indices) == 0:
            return []

        # Natural product halogen exclusion filter
        filtered_indices = matching_indices
        if not has_chlorine and self.has_cl.size == len(self.formulas):
            filtered_indices = filtered_indices[~self.has_cl[filtered_indices]]
        if not has_bromine and self.has_br.size == len(self.formulas):
            filtered_indices = filtered_indices[~self.has_br[filtered_indices]]
        if not has_chlorine and not has_bromine and self.has_f.size == len(self.formulas):
            filtered_indices = filtered_indices[~self.has_f[filtered_indices]]

        # Fallback to unfiltered matching if halogen filtering eliminates all candidates
        if len(filtered_indices) == 0:
            filtered_indices = matching_indices

        matched_items = []
        for idx in filtered_indices:
            f = self.formulas[idx]
            da_diff = diffs[idx]
            ppm_diff = (da_diff / target_mass) * 1e6
            conf = 1.0 / (1.0 + ppm_diff * 0.1)

            if target_carbon_estimate is not None and target_carbon_estimate > 0:
                c_diff = abs(self.carbon_counts[idx] - target_carbon_estimate)
                c_penalty = 1.0 / (1.0 + c_diff * 0.15)
                conf = conf * 0.6 + c_penalty * 0.4

            matched_items.append((f, float(conf)))

        matched_items.sort(key=lambda x: x[1], reverse=True)
        return matched_items[:max_results]
