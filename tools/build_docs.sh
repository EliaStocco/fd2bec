#!/usr/bin/env bash
set -e

# Build the mathematical documentation as a PDF.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m sphinx -M latexpdf docs/source docs/_build

# Make the PDF easy to find from the repository root.
ln -sfn docs/_build/latex/fd2bec_math.pdf fd2bec_math.pdf
