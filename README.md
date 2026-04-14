# Finite Differences to Born Effective Charges (fd2bec)
A python package to efficiently evaluate Born Effective Charges using Finite Differences.

# How to install
```bash
pyenv install 3.10 # it works also with 3.9 and 3.12
pyenv virtualenv 3.10 fd2bec
pyenv activate fd2bec
pyenv local fd2bec
pip install --upgrade pip
./tools/initialize.sh # for developers only, harmless anyway otherwise
pip install . # you can even use editable mode with 'pip install -e .'
```
# Testing
We would recommend running tests using
```bash
pytest --ff --nf -x
``` 