import pytest
import os
from pathlib import Path

test_dir = Path(__file__).parent.parent/"tests"
repo_root = Path(os.path.dirname(__file__))
structures_dir = repo_root / "structures"

@pytest.fixture(params=[1, 2, 3])
def structure(request):
    n = request.param
    file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
    return n, file_path