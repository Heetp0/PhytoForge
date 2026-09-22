"""Exact metric evaluation harness for CASMI InChIKey14 predictions."""

from typing import Any, Dict, Optional, Sequence, Tuple


def _normalize_key(val: Any) -> str:
    """Strip whitespace and convert to uppercase."""
    if val is None:
        return ""
    return str(val).strip().upper()


def _matches_ik14(target: str, cand: str) -> bool:
    """
    Check if a candidate matches the ground-truth target either by exact string
    equality (case-insensitive) or by 14-character InChIKey skeletal block prefix.
    """
    if not target or not cand:
        return False
    if target == cand:
        return True

    # Compare first block if hyphenated InChIKey
    t_block = target.split("-")[0] if "-" in target else target
    c_block = cand.split("-")[0] if "-" in cand else cand
    if t_block == c_block:
        return True

    # If both are at least 14 characters, check 14-char skeletal block
    if len(target) >= 14 and len(cand) >= 14:
        if target[:14] == cand[:14]:
            return True

    return False


def calculate_reciprocal_rank(
    ground_truth_ik14: str,
    predicted_slots: Sequence[str],
    k: int = 25,
) -> float:
    """
    Calculate reciprocal rank (1 / rank) for ground-truth InChIKey14 in top-k slots.

    Parameters
    ----------
    ground_truth_ik14 : str
        The true InChIKey or 14-character InChIKey prefix of the molecule.
    predicted_slots : Sequence[str]
        Ordered sequence of predicted candidate identifiers/InChIKeys.
    k : int, default=25
        The top-k cutoff. Any match after rank k yields 0.0.

    Returns
    -------
    float
        1.0 for rank 1, 0.5 for rank 2, ..., 1.0 / k for rank k; 0.0 if not found
        within top-k or if inputs are invalid / empty / k <= 0.
    """
    if k <= 0 or not ground_truth_ik14 or not predicted_slots:
        return 0.0

    target = _normalize_key(ground_truth_ik14)
    if not target:
        return 0.0

    limit = min(len(predicted_slots), k)
    for rank_idx in range(limit):
        cand = _normalize_key(predicted_slots[rank_idx])
        if _matches_ik14(target, cand):
            return 1.0 / float(rank_idx + 1)

    return 0.0


def evaluate_predictions(
    ground_truth: Dict[str, str],
    predictions: Dict[str, Sequence[str]],
    ks: Sequence[int] = (1, 5, 10, 25),
) -> Dict[str, float]:
    """
    Calculate aggregate MRR@k and Top-K accuracy across all query spectra.

    Parameters
    ----------
    ground_truth : Dict[str, str]
        Mapping from spectrum query ID to ground-truth InChIKey14.
    predictions : Dict[str, Sequence[str]]
        Mapping from spectrum query ID to ranked sequence of candidate predictions.
    ks : Sequence[int], default=(1, 5, 10, 25)
        Cutoff values for MRR and Top-K accuracy.

    Returns
    -------
    Dict[str, float]
        Dictionary containing 'mrr@{k}', 'top{k}_accuracy', and 'top_{k}_accuracy'
        for each k in ks.
    """
    metrics: Dict[str, float] = {}

    if not ground_truth:
        for k in ks:
            metrics[f"mrr@{k}"] = 0.0
            metrics[f"top{k}_accuracy"] = 0.0
            metrics[f"top_{k}_accuracy"] = 0.0
        return metrics

    total_queries = len(ground_truth)
    rr_sums = {k: 0.0 for k in ks}
    top_k_hits = {k: 0 for k in ks}

    for qid, target in ground_truth.items():
        preds = predictions.get(qid, ())
        for k in ks:
            rr = calculate_reciprocal_rank(target, preds, k=k)
            rr_sums[k] += rr
            if rr > 0.0:
                top_k_hits[k] += 1

    for k in ks:
        mrr_val = float(rr_sums[k] / total_queries)
        acc_val = float(top_k_hits[k] / total_queries)
        metrics[f"mrr@{k}"] = mrr_val
        metrics[f"top{k}_accuracy"] = acc_val
        metrics[f"top_{k}_accuracy"] = acc_val

    return metrics
