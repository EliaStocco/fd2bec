# Finite Differences to Born Effective Charges (fd2bec)
A python package to efficiently evaluate Born Effective Charges using Finite Differences.

# How to install
```bash
pyenv install 3.10             # it works with all versions from 3.9 to 3.14
pyenv virtualenv 3.10 fd2bec
pyenv activate fd2bec
pyenv local fd2bec
pip install --upgrade pip
./tools/initialize.sh          # for developers only, harmless anyway otherwise
pip install -e .               # only editable mode is fully tested so far
```
# Testing
We would recommend running tests using
```bash
pytest --ff --nf -x
``` 

# For developers
Run
```bash
ruff check . > ruff_check.txt
ruff check . --fix --unsafe-fixes > ruff_fix.txt
pylint  fd2bec \
--disable=invalid-name \
--disable=missing-function-docstring \
--disable=missing-module-docstring  > pylint.txt
ruff format .
find fd2bec tests -name "*.py" -exec sed -i 's/[ \t]*$//' {} + # to remove trailing-whitespace
```