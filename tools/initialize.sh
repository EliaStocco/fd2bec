#!/usr/bin/env bash
set -e

# Remove existing pyproject.toml if it exists (even if read-only)
if [ -f pyproject.toml ]; then
    chmod u+w pyproject.toml || true
    rm pyproject.toml
fi

# Generate scripts section
python tools/initialize.py > tools/scripts.toml

# Rebuild pyproject.toml
pylint --generate-toml-config > tools/pylint_pyproject.toml
cat tools/template.toml tools/scripts.toml tools/pylint_pyproject.toml > pyproject.toml

# Lock it down again
chmod 444 pyproject.toml

# Cleanup
rm tools/scripts.toml
rm tools/pylint_pyproject.toml
