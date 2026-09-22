"""
tests/test_kaggle_bundle.py

Comprehensive unit and integration test suite for:
- KaggleEnvironmentConfig (environment detection, asset resolution, fallback to None, overrides)
- build_kaggle_kernel (in-memory zip bundling, base64 encoding, AST parsing, subprocess execution)
- Standalone kernel execution (bootstrap extraction, 2-spectrum test run, submission validation)
- CLI entrypoint (python -m src.submission.kaggle_bundle)
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from src.submission.kaggle_bundle import (
    KaggleEnvironmentConfig,
    build_kaggle_kernel,
)
from src.submission.writer import validate_submission_file


class TestKaggleEnvironmentConfig:
    """Test KaggleEnvironmentConfig detection and asset resolution."""

    def test_environment_detection_local_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        with patch.object(Path, "is_dir", return_value=False):
            config = KaggleEnvironmentConfig()
            assert config.is_kaggle_environment() is False

    def test_environment_detection_force_kaggle(self) -> None:
        config_forced_true = KaggleEnvironmentConfig(force_kaggle=True)
        assert config_forced_true.is_kaggle_environment() is True

        config_forced_false = KaggleEnvironmentConfig(force_kaggle=False)
        assert config_forced_false.is_kaggle_environment() is False

    def test_environment_detection_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        config = KaggleEnvironmentConfig()
        assert config.is_kaggle_environment() is True

    def test_environment_detection_kaggle_input_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)

        def mock_is_dir(self: Path) -> bool:
            return "/kaggle/input" in str(self).replace("\\", "/")

        with patch.object(Path, "is_dir", mock_is_dir):
            config = KaggleEnvironmentConfig()
            assert config.is_kaggle_environment() is True

    def test_submission_output_resolution(self) -> None:
        local_config = KaggleEnvironmentConfig(force_kaggle=False)
        assert local_config.resolve_submission_output() == Path("dist/submission.csv")

        kaggle_config = KaggleEnvironmentConfig(force_kaggle=True)
        assert kaggle_config.resolve_submission_output() == Path("/kaggle/working/submission.csv")

        override_config = KaggleEnvironmentConfig(submission_output_path="custom/sub.csv")
        assert override_config.resolve_submission_output() == Path("custom/sub.csv")

    def test_asset_resolution_explicit_overrides(self, tmp_path: Path) -> None:
        db_file = tmp_path / "custom.db"
        db_file.touch()
        index_dir = tmp_path / "custom_index"
        index_dir.mkdir()
        frag_file = tmp_path / "frag.pkl"
        frag_file.touch()
        ranker_file = tmp_path / "model.txt"
        ranker_file.touch()
        test_file = tmp_path / "test.parquet"
        test_file.touch()

        config = KaggleEnvironmentConfig(
            candidate_db_path=db_file,
            spectral_index_path=index_dir,
            fragment_library_path=frag_file,
            meta_ranker_model_path=ranker_file,
            test_data_path=test_file,
        )

        assert config.resolve_candidate_db() == db_file
        assert config.resolve_spectral_index() == index_dir
        assert config.resolve_fragment_library() == frag_file
        assert config.resolve_meta_ranker_model() == ranker_file
        assert config.resolve_test_data() == test_file

    def test_asset_resolution_fallback_to_none(self, tmp_path: Path) -> None:
        # Search inside an empty directory
        config = KaggleEnvironmentConfig(base_input_dir=tmp_path)

        assert config.resolve_candidate_db() is None
        assert config.resolve_spectral_index() is None
        assert config.resolve_fragment_library() is None
        assert config.resolve_meta_ranker_model() is None
        assert config.resolve_test_data() is None

    def test_asset_resolution_directory_discovery(self, tmp_path: Path) -> None:
        db_dir = tmp_path / "candidates"
        db_dir.mkdir()
        db_file = db_dir / "candidates.db"
        db_file.touch()

        idx_dir = tmp_path / "dreams_index"
        idx_dir.mkdir()
        (idx_dir / "manifest.json").touch()

        frag_file = tmp_path / "neutral_loss_library.pkl"
        frag_file.touch()

        ranker_file = tmp_path / "lambdamart_ranker.txt"
        ranker_file.touch()

        test_file = tmp_path / "test.csv"
        test_file.touch()

        config = KaggleEnvironmentConfig(base_input_dir=tmp_path)

        assert config.resolve_candidate_db() == db_file
        assert config.resolve_spectral_index() == idx_dir
        assert config.resolve_fragment_library() == frag_file
        assert config.resolve_meta_ranker_model() == ranker_file
        assert config.resolve_test_data() == test_file

        resolved = config.resolve_all()
        assert resolved["candidate_db"] == db_file
        assert resolved["spectral_index"] == idx_dir
        assert resolved["fragment_library"] == frag_file
        assert resolved["meta_ranker_model"] == ranker_file
        assert resolved["test_data"] == test_file


class TestBuildKaggleKernel:
    """Test kernel building, AST validation, and standalone execution."""

    def test_build_kernel_creates_valid_ast(self, tmp_path: Path) -> None:
        out_file = tmp_path / "dist" / "kaggle_kernel.py"
        built_path = build_kaggle_kernel(output_file=out_file)

        assert built_path.exists()
        assert built_path.is_file()
        assert built_path.stat().st_size > 1000

        # Syntax check via AST parse
        code = built_path.read_text(encoding="utf-8")
        parsed = ast.parse(code)
        assert parsed is not None

    def test_build_kernel_missing_src_raises(self, tmp_path: Path) -> None:
        out_file = tmp_path / "kernel.py"
        non_existent_src = tmp_path / "non_existent_src"

        with pytest.raises(FileNotFoundError):
            build_kaggle_kernel(output_file=out_file, src_dir=non_existent_src)

    def test_kernel_help_subprocess_execution(self, tmp_path: Path) -> None:
        out_file = tmp_path / "kaggle_kernel.py"
        build_kaggle_kernel(output_file=out_file)

        result = subprocess.run(
            [sys.executable, str(out_file), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "PhytoForge Standalone Kaggle Kernel" in result.stdout

    def test_kernel_standalone_execution_on_test_spectra(self, tmp_path: Path) -> None:
        out_kernel = tmp_path / "kaggle_kernel.py"
        build_kaggle_kernel(output_file=out_kernel)

        # Create a 2-spectrum synthetic test dataset
        test_csv = tmp_path / "test_spectra.csv"
        df = pd.DataFrame({
            "id": ["SPEC_0001", "SPEC_0002"],
            "precursor_mz": [286.047, 300.0],
            "polarity": ["positive", "positive"],
            "peaks": ["100.0:10.0;200.0:20.0", "150.0:5.0;250.0:15.0"],
        })
        df.to_csv(test_csv, index=False)

        sub_csv = tmp_path / "submission.csv"

        result = subprocess.run(
            [
                sys.executable,
                str(out_kernel),
                "--input",
                str(test_csv),
                "--output",
                str(sub_csv),
                "--format",
                "inchikey14",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, f"Kernel failed with stderr: {result.stderr}"
        assert sub_csv.exists()

        # Validate generated submission file schema
        validate_submission_file(sub_csv, expected_ids=["SPEC_0001", "SPEC_0002"], output_format="inchikey14")


class TestKaggleBundleCLI:
    """Test CLI invocation of python -m src.submission.kaggle_bundle."""

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.submission.kaggle_bundle", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--output" in result.stdout

    def test_cli_build(self, tmp_path: Path) -> None:
        target_out = tmp_path / "dist" / "my_kernel.py"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.submission.kaggle_bundle",
                "--output",
                str(target_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"CLI build failed: {result.stderr}"
        assert target_out.exists()
        assert "successfully generated" in result.stdout
