"""Prepare and archive every structure in an extxyz dataset."""

import subprocess
import sys
from pathlib import Path

description = "Prepare a dataset for FHI-aims and archive each completed structure."


def main() -> int:
    """Run the packaged shell workflow with the supplied arguments."""
    script = Path(__file__).with_name("prepare_aims_dataset.sh")
    return subprocess.run(["bash", str(script), *sys.argv[1:]], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
