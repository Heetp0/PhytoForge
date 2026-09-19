"""Track 3: Bounded Generative De Novo Sampling."""

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union


@dataclass
class GenerativeCandidate:
    smiles: str
    inchikey14: str
    score: float
    source_tier: str = "track3_denovo"


class BoundedGenerativeEngine:
    """Track 3: Latency-bounded de novo generator with hard execution timeout and step limits."""

    def __init__(
        self,
        generator_fn: Optional[Callable] = None,
        timeout_seconds: float = 10.0,
        max_candidates: int = 5,
        max_steps: int = 50,
    ):
        self.generator_fn = generator_fn
        self.timeout_seconds = timeout_seconds
        self.max_candidates = max_candidates
        self.max_steps = max_steps

    def generate_with_timeout(
        self,
        spectrum_id: str,
        formula: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> List[GenerativeCandidate]:
        if self.generator_fn is None:
            return []

        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        )

        # Avoid context manager shutdown(wait=True) hang on timeout
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                self.generator_fn, spectrum_id, formula, max_steps=self.max_steps
            )
            raw_cands = future.result(timeout=effective_timeout)
        except (concurrent.futures.TimeoutError, Exception):
            executor.shutdown(wait=False, cancel_futures=True)
            return []
        else:
            executor.shutdown(wait=False)

        if not raw_cands:
            return []

        cands = []
        for item in raw_cands[: self.max_candidates]:
            if isinstance(item, GenerativeCandidate):
                cands.append(item)
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                cands.append(
                    GenerativeCandidate(
                        smiles=str(item[0]),
                        inchikey14=str(item[1]),
                        score=float(item[2]),
                    )
                )
        return cands
