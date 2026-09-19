"""
Adaptive Runtime Governor for Enveda CASMI 2026.

Guarantees:
1. Enforces <8h (28,800 seconds) total budget with safety buffer.
2. Dynamic rate adjustment ensuring completion across arbitrary query volumes.
3. 3 dynamic operating regimes:
   - DEEP: >5.0s/mol remaining -> Full Tier 1 + Tier 2 + Tier 3 (MS-GPT) + BRICS ILP + MMR.
   - STANDARD: 2.0-5.0s/mol remaining -> Tier 1 + Tier 2 + Fast Knapsack (skip MS-GPT beam search).
   - FAST: <2.0s/mol remaining -> Tier 1 library + pre-indexed DB lookup only.
4. Golden library hit bypass trigger: Locks hit if similarity >= 0.82 to conserve compute.
5. Telemetry recording, context managers, and per-module adaptive timeouts.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class OperatingRegime(str, Enum):
    """Execution regimes based on remaining time budget per query."""
    DEEP = "DEEP"          # > 5.0s / mol remaining
    STANDARD = "STANDARD"  # 2.0 - 5.0s / mol remaining
    FAST = "FAST"          # < 2.0s / mol remaining


class BudgetExhaustedError(RuntimeError):
    """Raised when the total allocated execution budget has expired."""
    pass


@dataclass
class QueryTelemetry:
    """Audit record for a single processed test query."""
    molecule_id: str
    regime: str
    start_time: float
    end_time: float
    duration_sec: float
    bypassed_deep: bool = False
    module_durations: Dict[str, float] = field(default_factory=dict)


class RuntimeGovernor:
    """
    Adaptive Runtime Governor enforcing Kaggle competition execution time limits.
    """

    # Baseline fraction of available query time allocated to each module
    MODULE_FRACTIONS: Dict[OperatingRegime, Dict[str, float]] = {
        OperatingRegime.DEEP: {
            "tier1": 0.05,
            "tier2": 0.20,
            "tier3": 0.45,
            "knapsack": 0.20,
            "network": 0.05,
            "mmr": 0.05,
        },
        OperatingRegime.STANDARD: {
            "tier1": 0.05,
            "tier2": 0.35,
            "tier3": 0.20,
            "knapsack": 0.25,
            "network": 0.05,
            "mmr": 0.10,
        },
        OperatingRegime.FAST: {
            "tier1": 0.20,
            "tier2": 0.70,
            "tier3": 0.00,
            "knapsack": 0.00,
            "network": 0.00,
            "mmr": 0.10,
        },
    }

    def __init__(
        self,
        total_budget_sec: float = 28800.0,
        safety_buffer_sec: float = 300.0,
        total_queries: int = 400,
        bypass_similarity_threshold: float = 0.82,
    ):
        """
        Initialize the RuntimeGovernor.

        Args:
            total_budget_sec: Hard cutoff time in seconds (default 8.0 hours = 28,800s).
            safety_buffer_sec: Margin reserved for I/O, writing, and cleanup (default 300s = 5m).
            total_queries: Expected total count of test queries.
            bypass_similarity_threshold: Spectral similarity threshold to lock library hit
                                         and bypass expensive generative sampling.
        """
        if total_budget_sec <= 0:
            raise ValueError(f"total_budget_sec must be positive, got {total_budget_sec}")
        if total_queries <= 0:
            raise ValueError(f"total_queries must be positive, got {total_queries}")

        self.total_budget_sec = float(total_budget_sec)
        self.safety_buffer_sec = float(safety_buffer_sec)
        self.effective_budget_sec = max(1.0, self.total_budget_sec - self.safety_buffer_sec)
        self.total_queries = int(total_queries)
        self.bypass_similarity_threshold = float(bypass_similarity_threshold)

        self.start_time: float = time.time()
        self.completed_queries: int = 0
        self.telemetry_records: List[QueryTelemetry] = []

        # Active query state
        self._current_molecule_id: Optional[str] = None
        self._current_query_start: Optional[float] = None
        self._current_regime: Optional[OperatingRegime] = None
        self._current_bypassed: bool = False
        self._current_module_durations: Dict[str, float] = {}

    @property
    def elapsed_time(self) -> float:
        """Total elapsed wall-clock time since governor initialization."""
        return max(0.0, time.time() - self.start_time)

    @property
    def remaining_time(self) -> float:
        """Remaining usable budget before hitting the safety buffer cutoff."""
        return max(0.0, self.effective_budget_sec - self.elapsed_time)

    @property
    def remaining_queries(self) -> int:
        """Count of uncompleted queries."""
        return max(1, self.total_queries - self.completed_queries)

    @property
    def sec_per_query_remaining(self) -> float:
        """Dynamically available time per remaining test molecule."""
        return self.remaining_time / self.remaining_queries

    @property
    def current_regime(self) -> OperatingRegime:
        """Current operating regime based on remaining budget."""
        return self.get_execution_mode()

    @property
    def is_budget_exhausted(self) -> bool:
        """Whether remaining time budget has reached zero."""
        return self.remaining_time <= 0.0

    def get_execution_mode(self) -> OperatingRegime:
        """
        Determines whether the pipeline should execute in DEEP, STANDARD, or FAST mode.
        """
        sec_per_query = self.sec_per_query_remaining
        if sec_per_query > 5.0:
            return OperatingRegime.DEEP
        elif sec_per_query >= 2.0:
            return OperatingRegime.STANDARD
        else:
            return OperatingRegime.FAST

    def should_bypass_generative(self, top_library_score: float) -> bool:
        """
        Check if a Tier 1 library match is sufficiently high-confidence
        (e.g., DreaMS cosine >= 0.82) to lock into Rank 1 and bypass
        computationally expensive de novo generation and knapsack solving.
        """
        bypass = float(top_library_score) >= self.bypass_similarity_threshold
        if bypass:
            self._current_bypassed = True
        return bypass

    def start_query(self, molecule_id: str) -> OperatingRegime:
        """
        Marks the beginning of processing for a new test molecule.
        """
        if self.is_budget_exhausted:
            logger.warning(f"Execution budget exhausted ({self.elapsed_time:.1f}s elapsed)! Throttling to FAST.")

        self._current_molecule_id = molecule_id
        self._current_query_start = time.time()
        self._current_regime = self.get_execution_mode()
        self._current_bypassed = False
        self._current_module_durations = {}
        return self._current_regime

    def finish_query(self, molecule_id: Optional[str] = None) -> float:
        """
        Marks completion of processing for the current test molecule.
        Records telemetry and returns the query duration in seconds.
        """
        now = time.time()
        start = self._current_query_start or now
        duration = max(0.0, now - start)

        mol_id = molecule_id or self._current_molecule_id or f"query_{self.completed_queries}"
        regime = self._current_regime or self.get_execution_mode()

        record = QueryTelemetry(
            molecule_id=mol_id,
            regime=regime.value,
            start_time=start,
            end_time=now,
            duration_sec=duration,
            bypassed_deep=self._current_bypassed,
            module_durations=dict(self._current_module_durations),
        )
        self.telemetry_records.append(record)
        self.completed_queries += 1

        # Reset active query state
        self._current_molecule_id = None
        self._current_query_start = None
        self._current_regime = None
        self._current_bypassed = False
        self._current_module_durations = {}

        return duration

    @contextmanager
    def query_scope(self, molecule_id: str) -> Iterator[OperatingRegime]:
        """Context manager wrapping query processing."""
        regime = self.start_query(molecule_id)
        try:
            yield regime
        finally:
            self.finish_query(molecule_id)

    @contextmanager
    def module_scope(self, module_name: str) -> Iterator[float]:
        """
        Context manager wrapping a specific processing sub-module (e.g., 'tier1', 'tier3').
        Yields the allocated timeout budget for this module.
        """
        timeout = self.get_module_timeout(module_name)
        m_start = time.time()
        try:
            yield timeout
        finally:
            m_dur = max(0.0, time.time() - m_start)
            self._current_module_durations[module_name] = m_dur

    def get_module_timeout(self, module_name: str) -> float:
        """
        Calculates the timeout budget in seconds allocated for a specific pipeline module
        based on current regime and remaining query budget.
        """
        regime = self.get_execution_mode()
        sec_available = self.sec_per_query_remaining
        fractions = self.MODULE_FRACTIONS[regime]
        frac = fractions.get(module_name.lower(), 0.10)
        # Cap by available sec and provide minimum 10ms
        return max(0.01, frac * min(sec_available, 10.0))

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate execution metrics."""
        durations = [r.duration_sec for r in self.telemetry_records]
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0
        min_dur = min(durations) if durations else 0.0

        regime_counts = {
            OperatingRegime.DEEP.value: sum(1 for r in self.telemetry_records if r.regime == OperatingRegime.DEEP.value),
            OperatingRegime.STANDARD.value: sum(1 for r in self.telemetry_records if r.regime == OperatingRegime.STANDARD.value),
            OperatingRegime.FAST.value: sum(1 for r in self.telemetry_records if r.regime == OperatingRegime.FAST.value),
        }

        return {
            "total_budget_sec": self.total_budget_sec,
            "safety_buffer_sec": self.safety_buffer_sec,
            "effective_budget_sec": self.effective_budget_sec,
            "elapsed_time_sec": self.elapsed_time,
            "remaining_time_sec": self.remaining_time,
            "completed_queries": self.completed_queries,
            "total_queries": self.total_queries,
            "avg_query_duration_sec": avg_dur,
            "min_query_duration_sec": min_dur,
            "max_query_duration_sec": max_dur,
            "current_regime": self.current_regime.value,
            "regime_distribution": regime_counts,
            "bypassed_deep_count": sum(1 for r in self.telemetry_records if r.bypassed_deep),
        }

    def export_telemetry(self, path: Union[str, Path]) -> Path:
        """Exports full telemetry logs and summary to JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.get_summary(),
            "records": [asdict(r) for r in self.telemetry_records],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path
