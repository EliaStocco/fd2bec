import os
from pathlib import Path

test_dir = Path(__file__).parent.parent/"tests"
repo_root = Path(os.path.dirname(__file__))
structures_dir = repo_root / "structures"