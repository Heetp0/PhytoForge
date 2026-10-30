"""
Core Process Telemetry & Formatting Utility for Enveda CASMI 2026.

Provides:
- get_process_memory_gb: High-efficiency RSS probe via /proc/self/statm with psutil fallback.
- format_heartbeat: Single-line unbuffered query progress telemetry log.
- format_banner: Multi-line runtime startup banner with resolved assets.
- format_completion: Multi-line execution summary with speed, memory, and hit metrics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union

# Cache system page size at module scope
try:
    _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
except Exception:
    _PAGE_SIZE = 4096


def get_process_memory_gb() -> float:
    """
    Get current process Resident Set Size (RSS) memory consumption in gigabytes (GB).

    Reads /proc/self/statm on Linux (index 1 is RSS in pages), multiplies by _PAGE_SIZE,
    and converts to GB. Falls back to psutil.Process().memory_info().rss if /proc is
    unavailable, and returns 0.0 if any error occurs.
    """
    try:
        try:
            with open("/proc/self/statm", "r") as f:
                content = f.read().strip()
                if content:
                    parts = content.split()
                    if len(parts) >= 2:
                        rss_pages = int(parts[1])
                        return float((rss_pages * _PAGE_SIZE) / (1024.0 ** 3))
        except (OSError, IOError, ValueError, IndexError):
            pass

        import psutil

        proc = psutil.Process()
        return float(proc.memory_info().rss / (1024.0 ** 3))
    except Exception:
        return 0.0


def format_heartbeat(
    idx: int,
    total: int,
    query_id: str,
    top_cosine: float,
    matched_adduct: str,
    is_rank1: bool,
    hits_count: int,
    sec_per_spec: float,
    ram_gb: float,
    eta_sec: float,
) -> str:
    """
    Format single-line unbuffered query progress heartbeat.

    Format:
    [001/100] (  1.0%) Query: spec_001 | Top Cos: 0.8523 ([M+H]+) | Match: Rank 1 | Hits: 1/1 (100.0%) | 1.20s/mol | RAM: 2.50 GB | ETA: 45s
    """
    pct = (idx / total * 100.0) if total > 0 else 0.0
    hit_pct = (hits_count / idx * 100.0) if idx > 0 else 0.0
    status_str = "Rank 1" if is_rank1 else "DB/Heuristic"
    adduct_str = f" ({matched_adduct})" if matched_adduct else ""
    eta_str = f"{eta_sec / 60.0:.1f}m" if eta_sec >= 60.0 else f"{max(0.0, eta_sec):.0f}s"

    return (
        f"[{idx:03d}/{total:03d}] ({pct:5.1f}%) Query: {query_id} | "
        f"Top Cos: {top_cosine:.4f}{adduct_str} | Match: {status_str} | "
        f"Hits: {hits_count}/{idx} ({hit_pct:.1f}%) | "
        f"{sec_per_spec:.2f}s/mol | RAM: {ram_gb:.2f} GB | ETA: {eta_str}"
    )


def format_banner(assets: Dict[str, Any], total_queries: int) -> str:
    """
    Format multi-line pipeline initialization banner string.
    """
    lines = [
        "=" * 80,
        ">>> PhytoForge v1.0.6 [Live Telemetry Active] - Initializing Pipeline",
        f">>> Target Queries: {total_queries}",
        ">>> Resolved Assets:",
    ]
    for key, val in sorted(assets.items()):
        name = getattr(val, "name", val) if val is not None else "none"
        lines.append(f"    - {key}: {name}")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_completion(
    total_queries: int,
    elapsed_sec: float,
    authentic_hits: int,
    peak_ram_gb: float,
    output_path: Union[str, Path],
) -> str:
    """
    Format multi-line pipeline completion summary string.
    """
    speed = elapsed_sec / max(1, total_queries)
    hit_pct = (authentic_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    lines = [
        "=" * 80,
        ">>> PhytoForge Pipeline Run Completed Successfully",
        f"    - Total Queries:   {total_queries}",
        f"    - Total Time:      {elapsed_sec:.2f}s ({elapsed_sec / 60.0:.2f} min)",
        f"    - Average Speed:   {speed:.2f}s/spec",
        f"    - Authentic Hits:  {authentic_hits}/{total_queries} ({hit_pct:.1f}%)",
        f"    - Peak Memory:     {peak_ram_gb:.2f} GB",
        f"    - Submission File: {output_path}",
        "=" * 80,
    ]
    return "\n".join(lines)


__all__ = [
    "get_process_memory_gb",
    "format_heartbeat",
    "format_banner",
    "format_completion",
]
