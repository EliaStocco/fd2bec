"""Recreate one structure folder's prepare_aims outputs."""

import subprocess
import sys
from pathlib import Path

description = "Recreate a structure folder from start.extxyz."


def main() -> int:
    """Run the packaged rerun shell workflow with the supplied arguments."""
    script = Path(__file__).with_name("rerun_prepare_aims.sh")
    return subprocess.run(["bash", str(script), *sys.argv[1:]], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
