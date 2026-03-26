import os
from pathlib import Path

test_dir = Path(__file__).parent.parent/"tests"

# Get the root of the repo relative to this test file
repo_root = Path(os.path.dirname(os.path.dirname(__file__)))

structures_dir = repo_root / "structures"