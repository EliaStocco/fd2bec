# Finite Differences to Born Effective Charges (fd2bec)
A python package to efficiently evaluate Born Effective Charges using Finite Differences.

Please do the following:
 - rename the folder `NAME` with the chosen name of the package
 - replace `DESCRIPTION` with an actual description in `tools/template.toml`
 - replace `NAME` with the chosen name of the package in:
    - `tools/template.toml`
    - this file below
    - `tools/initialize.py` 
    - `.github/workflows/pytest.yml`

# How to install
```bash
pyenv install 3.10 -y
pyenv virtualenv 3.10 fd2bec
pyenv activate fd2bec
pyenv local fd2bec
python -m pip install --upgrade pip
./tools/initialize.sh
pip install -e .
```


