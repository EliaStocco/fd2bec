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

echo "8) Running Pylint with specific checks disabled..."
pylint  fd2bec \
--disable=invalid-name \
--disable=missing-function-docstring \
--disable=missing-module-docstring  \
--disable=too-many-locals \
--disable=too-many-statements

echo "9) Running Pytest with coverage report..."
pytest --cov=fd2bec --cov-report=term-missing

echo "✅ All formatting complete!"
