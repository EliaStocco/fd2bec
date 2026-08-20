#!/usr/bin/env bash
set -e

# Refresh the generated code maps and build the documentation as a PDF.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 tools/generate_code_map.py
python3 -m sphinx -M latexpdf docs/source docs/_build

# Make the PDF easy to find from the repository root.
ln -sfn docs/_build/latex/fd2bec_math.pdf fd2bec_math.pdf
