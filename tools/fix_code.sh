#!/usr/bin/env bash

set -e

echo "🔧 Running Ruff (lint + auto-fix)..."
ruff check . --fix

echo "🎨 Running Ruff format..."
ruff format .

echo "🎨 Running Black..."
black .

echo "🧹 Fixing end-of-file issues..."
pre-commit run end-of-file-fixer --all-files

echo "🧹 Fixing trailing whitespace..."
pre-commit run trailing-whitespace --all-files

echo "✅ All formatting complete!"