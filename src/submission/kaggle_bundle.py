"""
Kaggle Offline Bundle Generator and Environment Adapter for Enveda CASMI 2026.

Provides:
- KaggleEnvironmentConfig: Auto-detects /kaggle/input vs local runtime and resolves asset paths
  (candidate SQLite database, spectral index, fragment library, meta-ranker model, test dataset,
  and submission output location).
- build_kaggle_kernel: Compiles an all-in-one standalone runnable submission kernel (kaggle_kernel.py)
  with embedded base64 in-memory zip archive of src/ modules, dynamic runtime governor, and
  submission validation under Kaggle's 16 GB RAM and 21.3s/spectrum limits.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import zipfile


class KaggleEnvironmentConfig:
    """
    Environment adapter resolving offline competition asset paths across Kaggle and local environments.
    """

    def __init__(
        self,
        candidate_db_path: Optional[Union[str, Path]] = None,
        spectral_index_path: Optional[Union[str, Path]] = None,
        fragment_library_path: Optional[Union[str, Path]] = None,
        meta_ranker_model_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None,
        submission_output_path: Optional[Union[str, Path]] = None,
        force_kaggle: Optional[bool] = None,
        base_input_dir: Optional[Union[str, Path]] = None,
    ):
        self.candidate_db_path = Path(candidate_db_path) if candidate_db_path is not None else None
        self.spectral_index_path = Path(spectral_index_path) if spectral_index_path is not None else None
        self.fragment_library_path = Path(fragment_library_path) if fragment_library_path is not None else None
        self.meta_ranker_model_path = Path(meta_ranker_model_path) if meta_ranker_model_path is not None else None
        self.test_data_path = Path(test_data_path) if test_data_path is not None else None
        self.submission_output_path = Path(submission_output_path) if submission_output_path is not None else None
        self.force_kaggle = force_kaggle
        self.base_input_dir = Path(base_input_dir) if base_input_dir is not None else None

    def is_kaggle_environment(self) -> bool:
        """
        Detect whether running in Kaggle container environment.
        Checked via force_kaggle override, KAGGLE_KERNEL_RUN_TYPE env var, or /kaggle/input directory.
        """
        if self.force_kaggle is not None:
            return bool(self.force_kaggle)
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
            return True
        try:
            return Path("/kaggle/input").is_dir()
        except (PermissionError, OSError):
            return False

    def _get_search_roots(self) -> List[Path]:
        """Return search directories in priority order based on environment."""
        if self.base_input_dir is not None and self.base_input_dir.exists():
            return [self.base_input_dir]

        if self.is_kaggle_environment():
            kaggle_input = Path("/kaggle/input")
            return [kaggle_input] if kaggle_input.exists() else []

        # Local search locations
        roots = [
            Path.cwd(),
            Path.cwd() / "data",
            Path.cwd() / "models",
            Path.cwd() / "indices",
            Path.cwd() / "dist",
        ]
        return [r for r in roots if r.exists()]

    def resolve_candidate_db(self) -> Optional[Path]:
        """
        Resolve path to candidates SQLite database.
        Returns explicit path if provided, otherwise searches candidate databases (*.db).
        Returns None if not found.
        """
        if self.candidate_db_path is not None:
            return self.candidate_db_path if self.candidate_db_path.exists() else None

        for root in self._get_search_roots():
            # Check exact candidate database names
            for target_name in ("candidates.db", "compound_candidates.db"):
                exact = root / target_name
                if exact.is_file():
                    return exact

            # Recursive search for candidates*.db or *.db
            try:
                for match in root.rglob("*.db"):
                    if match.is_file() and not match.name.startswith("."):
                        return match
            except (PermissionError, OSError):
                continue

        return None

    def resolve_spectral_index(self) -> Optional[Path]:
        """
        Resolve path to DreaMS spectral embedding index directory.
        Looks for manifest.json and embeddings.npy. Returns None if not found.
        """
        if self.spectral_index_path is not None:
            return self.spectral_index_path if self.spectral_index_path.exists() else None

        for root in self._get_search_roots():
            # Direct check if root itself is a spectral index
            if (root / "manifest.json").is_file():
                return root

            # Check subdirectories
            try:
                for manifest in root.rglob("manifest.json"):
                    if manifest.is_file():
                        return manifest.parent
            except (PermissionError, OSError):
                continue

        return None

    def resolve_fragment_library(self) -> Optional[Path]:
        """
        Resolve path to precomputed fragment / neutral loss library (*.pkl).
        Returns None if not found.
        """
        if self.fragment_library_path is not None:
            return self.fragment_library_path if self.fragment_library_path.exists() else None

        for root in self._get_search_roots():
            try:
                for pkl in root.rglob("*.pkl"):
                    name_lower = pkl.name.lower()
                    if "fragment" in name_lower or "loss" in name_lower or "neutral" in name_lower:
                        if pkl.is_file():
                            return pkl
            except (PermissionError, OSError):
                continue

            # Fallback to any .pkl file in root
            try:
                for pkl in root.rglob("*.pkl"):
                    if pkl.is_file() and not pkl.name.startswith("."):
                        return pkl
            except (PermissionError, OSError):
                continue

        return None

    def resolve_meta_ranker_model(self) -> Optional[Path]:
        """
        Resolve path to trained LightGBM LambdaMART ranker booster (*.txt).
        Returns None if not found.
        """
        if self.meta_ranker_model_path is not None:
            return self.meta_ranker_model_path if self.meta_ranker_model_path.exists() else None

        for root in self._get_search_roots():
            try:
                for txt in root.rglob("*.txt"):
                    name_lower = txt.name.lower()
                    if ("ranker" in name_lower or "lambdamart" in name_lower or "booster" in name_lower) and txt.is_file():
                        return txt
            except (PermissionError, OSError):
                continue

            # Check model.txt or lightgbm.txt
            for name in ("model.txt", "lightgbm_model.txt", "ranker.txt"):
                target = root / name
                if target.is_file():
                    return target

        return None

    def resolve_test_data(self) -> Optional[Path]:
        """
        Resolve path to test query dataset (test.parquet or test.csv).
        Returns None if not found.
        """
        if self.test_data_path is not None:
            return self.test_data_path if self.test_data_path.exists() else None

        for root in self._get_search_roots():
            # Check for direct test.parquet or test.csv
            for name in ("test.parquet", "test.csv"):
                direct = root / name
                if direct.is_file():
                    return direct

            # Recursive search for test.parquet or test.csv
            try:
                for match in root.rglob("test.parquet"):
                    if match.is_file():
                        return match
                for match in root.rglob("test.csv"):
                    if match.is_file():
                        return match
            except (PermissionError, OSError):
                continue

        return None

    def resolve_submission_output(self) -> Path:
        """
        Resolve destination path for submission CSV.
        Defaults to /kaggle/working/submission.csv in Kaggle, dist/submission.csv locally.
        """
        if self.submission_output_path is not None:
            return self.submission_output_path

        if self.is_kaggle_environment():
            return Path("/kaggle/working/submission.csv")

        return Path("dist/submission.csv")

    def resolve_all(self) -> Dict[str, Optional[Path]]:
        """Return summary of all resolved assets."""
        return {
            "is_kaggle": Path("/kaggle/input") if self.is_kaggle_environment() else None,
            "candidate_db": self.resolve_candidate_db(),
            "spectral_index": self.resolve_spectral_index(),
            "fragment_library": self.resolve_fragment_library(),
            "meta_ranker_model": self.resolve_meta_ranker_model(),
            "test_data": self.resolve_test_data(),
            "submission_output": self.resolve_submission_output(),
        }


KERNEL_TEMPLATE = '''"""
Standalone Kaggle Submission Kernel for Enveda CASMI 2026.
Compiled by PhytoForge Kaggle Offline Bundle Generator.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import os
from pathlib import Path
import sys
import time
import zipfile

# Embedded source archive of PhytoForge src/ packages
EMBEDDED_SOURCE_B64 = """{embedded_b64}"""


def _bootstrap_source(target_dir: Path | None = None) -> Path:
    """Extract embedded source archive to target directory and insert into sys.path."""
    if target_dir is None:
        target_dir = Path.cwd() / ".kaggle_bundle_src"
    target_dir = Path(target_dir)
    target_src = target_dir / "src"
    if not (target_src / "pipeline.py").exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_bytes = base64.b64decode(EMBEDDED_SOURCE_B64.strip())
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            zf.extractall(target_dir)
    target_str = str(target_dir.resolve())
    if target_str not in sys.path:
        sys.path.insert(0, target_str)
    return target_dir


# Execute source bootstrap
_bootstrap_source()

import pandas as pd
from src.pipeline import CASMIOmegaPipeline
from src.submission.kaggle_bundle import KaggleEnvironmentConfig
from src.submission.runtime_governor import DynamicRuntimeGovernor
from src.submission.writer import validate_submission_file


def run_kaggle_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_format: str = "inchikey14",
    db_path: str | Path | None = None,
    spectral_index_path: str | Path | None = None,
    fragment_library_path: str | Path | None = None,
    meta_ranker_path: str | Path | None = None,
    governor_limit_per_spec: float = 21.3,
    governor_reserve: float = 480.0,
    force_kaggle: bool | None = None,
) -> Path:
    """Run CASMIOmegaPipeline with Kaggle asset resolution and dynamic governor."""
    config = KaggleEnvironmentConfig(
        candidate_db_path=db_path,
        spectral_index_path=spectral_index_path,
        fragment_library_path=fragment_library_path,
        meta_ranker_model_path=meta_ranker_path,
        test_data_path=input_path,
        submission_output_path=output_path,
        force_kaggle=force_kaggle,
    )

    resolved_test = config.resolve_test_data()
    if resolved_test is None:
        raise FileNotFoundError(
            "Test data file not found. Provide --input or ensure test.parquet/test.csv is accessible."
        )

    resolved_output = config.resolve_submission_output()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    if resolved_test.suffix == ".parquet":
        test_df = pd.read_parquet(resolved_test)
    else:
        try:
            test_df = pd.read_csv(resolved_test)
        except UnicodeDecodeError:
            test_df = pd.read_csv(resolved_test, encoding="latin-1")

    total_specs = len(test_df)
    total_budget = governor_limit_per_spec * total_specs if total_specs > 0 else 32400.0

    resolved_db = config.resolve_candidate_db()
    resolved_index = config.resolve_spectral_index()
    resolved_frag = config.resolve_fragment_library()

    frag_lib = None
    if resolved_frag is not None:
        try:
            import pickle
            from src.reranking.fragment_library import NeutralLossLibrary
            with open(resolved_frag, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, NeutralLossLibrary):
                    frag_lib = loaded.library
                elif isinstance(loaded, dict):
                    frag_lib = loaded
        except Exception:
            frag_lib = None

    with CASMIOmegaPipeline(
        db_path=resolved_db,
        spectral_index_path=resolved_index,
        fragment_library=frag_lib,
        governor_budget_seconds=total_budget,
    ) as pipeline:
        pipeline.run(
            test_df=test_df,
            output_path=resolved_output,
            output_format=output_format,
        )

    # Validate output submission file
    validate_submission_file(resolved_output, output_format=output_format)
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhytoForge Standalone Kaggle Kernel for Enveda CASMI 2026"
    )
    parser.add_argument("--input", type=str, default=None, help="Path to input test parquet or csv")
    parser.add_argument("--output", type=str, default=None, help="Path to output submission CSV")
    parser.add_argument(
        "--format",
        type=str,
        default="inchikey14",
        choices=["inchikey14", "smiles"],
        help="Submission output format",
    )
    parser.add_argument("--db-path", type=str, default=None, help="Path to candidate SQLite database")
    parser.add_argument("--spectral-index-path", type=str, default=None, help="Path to spectral index")
    parser.add_argument("--fragment-library-path", type=str, default=None, help="Path to fragment library")
    parser.add_argument("--meta-ranker-path", type=str, default=None, help="Path to meta-ranker model")
    parser.add_argument("--governor", type=float, default=21.3, help="Max budget per spectrum in seconds")
    parser.add_argument("--reserve", type=float, default=480.0, help="Governor reserve seconds")
    parser.add_argument(
        "--force-kaggle",
        action="store_true",
        default=None,
        help="Force Kaggle environment detection",
    )

    args = parser.parse_args()
    try:
        out = run_kaggle_pipeline(
            input_path=args.input,
            output_path=args.output,
            output_format=args.format,
            db_path=args.db_path,
            spectral_index_path=args.spectral_index_path,
            fragment_library_path=args.fragment_library_path,
            meta_ranker_path=args.meta_ranker_path,
            governor_limit_per_spec=args.governor,
            governor_reserve=args.reserve,
            force_kaggle=args.force_kaggle,
        )
        print(f"Kaggle kernel run completed successfully. Submission output: {{out}}")
    except Exception as exc:
        print(f"Kaggle kernel execution failed: {{exc}}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def build_kaggle_kernel(
    output_file: Union[str, Path],
    src_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Compile an all-in-one runnable standalone submission kernel kaggle_kernel.py.

    Gathers all .py files under src/ into an in-memory zip archive, base64 encodes it,
    and inlines it into the standalone kernel template. The kernel bootstraps into
    .kaggle_bundle_src and preserves 100% namespace fidelity.

    Parameters
    ----------
    output_file : Union[str, Path]
        Path to output generated Python script.
    src_dir : Optional[Union[str, Path]]
        Path to src directory. Defaults to repository src/.

    Returns
    -------
    Path
        Resolved path to the generated script.
    """
    out_path = Path(output_file).resolve()

    if src_dir is not None:
        src_path = Path(src_dir).resolve()
    else:
        # Resolve repository src directory relative to this file
        src_path = Path(__file__).resolve().parent.parent

    if not src_path.exists() or not src_path.is_dir():
        raise FileNotFoundError(f"Source directory '{src_path}' does not exist or is not a directory.")

    # Build in-memory zip archive of all .py files under src_path
    buf = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for py_file in sorted(src_path.rglob("*.py")):
            # Skip caches
            if "__pycache__" in py_file.parts:
                continue
            rel_path = py_file.relative_to(src_path)
            arcname = f"src/{rel_path.as_posix()}"
            zf.write(py_file, arcname=arcname)
            file_count += 1

    if file_count == 0:
        raise ValueError(f"No Python source files found under '{src_path}' to bundle.")

    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

    # Format kernel script
    kernel_code = KERNEL_TEMPLATE.format(embedded_b64=b64_str)

    # Validate Python syntax via AST
    try:
        ast.parse(kernel_code, filename=str(out_path))
    except SyntaxError as exc:
        raise ValueError(f"Generated standalone kernel has syntax error: {exc}") from exc

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kernel_code, encoding="utf-8")

    return out_path


def main() -> None:
    """CLI entrypoint for Kaggle offline bundle generator."""
    parser = argparse.ArgumentParser(
        description="Compile standalone Kaggle offline submission kernel for Enveda CASMI 2026."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="dist/kaggle_kernel.py",
        help="Target output file path for standalone kernel (default: dist/kaggle_kernel.py)",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default=None,
        help="Source directory containing src/ package (default: auto-detected repository src/)",
    )

    args = parser.parse_args()

    try:
        out_path = build_kaggle_kernel(
            output_file=args.output,
            src_dir=args.src_dir,
        )
        size_kb = out_path.stat().st_size / 1024.0
        print(f"Kaggle standalone kernel successfully generated at: {out_path} ({size_kb:.1f} KB)")
        sys.exit(0)
    except Exception as exc:
        print(f"Failed to build Kaggle bundle: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
