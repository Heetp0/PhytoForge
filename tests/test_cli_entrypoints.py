import subprocess
import sys
from pathlib import Path

def test_pipeline_help():
    result = subprocess.run([sys.executable, '-m', 'src.submission.pipeline', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "PhytoForge inference pipeline runner" in result.stdout

def test_validator_help():
    result = subprocess.run([sys.executable, '-m', 'src.submission.validator', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "PhytoForge submission integrity validator" in result.stdout

def test_pipeline_missing_args():
    result = subprocess.run([sys.executable, '-m', 'src.submission.pipeline'], capture_output=True, text=True)
    assert result.returncode != 0
    assert "the following arguments are required: --input, --output" in result.stderr

def test_validator_missing_args():
    result = subprocess.run([sys.executable, '-m', 'src.submission.validator'], capture_output=True, text=True)
    assert result.returncode != 0
    assert "the following arguments are required: --submission" in result.stderr

def test_pipeline_nonexistent_input():
    result = subprocess.run([sys.executable, '-m', 'src.submission.pipeline', '--input', 'doesnotexist.parquet', '--output', 'out.csv'], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Error: input file 'doesnotexist.parquet' not found." in result.stderr

def test_validator_nonexistent_submission():
    result = subprocess.run([sys.executable, '-m', 'src.submission.validator', '--submission', 'doesnotexist.csv'], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Error: submission file 'doesnotexist.csv' not found." in result.stderr

def test_pipeline_importable():
    import src.submission.pipeline
    assert hasattr(src.submission.pipeline, 'main')

def test_validator_importable():
    import src.submission.validator
    assert hasattr(src.submission.validator, 'main')

def test_pipeline_main_exists():
    from src.submission.pipeline import main
    assert callable(main)

def test_validator_main_exists():
    from src.submission.validator import main
    assert callable(main)
