#!/usr/bin/env bash

set -e

echo "1) Removing trailing whitespace from all Python files..."
find fd2bec tests -name "*.py" -exec sed -i 's/[ \t]*$//' {} +

echo "2) Running Ruff (lint + auto-fix)..."
ruff check . --fix

echo "3) Running Ruff format..."
ruff format .

echo "4) Running Black..."
black .

echo "4) Running isort..."
isort .

echo "6) Fixing end-of-file issues..."
pre-commit run end-of-file-fixer --all-files

echo "7) Fixing trailing whitespace..."
pre-commit run trailing-whitespace --all-files

echo "✅ All formatting complete!"
