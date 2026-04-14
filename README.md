# Finite Differences to Born Effective Charges (fd2bec)
A python package to efficiently evaluate Born Effective Charges using Finite Differences.

# How to install
```bash
pyenv install 3.10
pyenv virtualenv 3.10 fd2bec
pyenv activate fd2bec
pyenv local fd2bec
pip install --upgrade pip
./tools/initialize.sh # for developers only, harmless anyway otherwise
pip install -e .
```
# Testing
We would recommend running test using
```bash
pytest --ff --nf -x
``` 