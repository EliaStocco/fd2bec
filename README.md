# Template
Template repository for a python package

Please do the following:
 - rename the folder `NAME` with the chosen name of the package
 - replace `DESCRIPTION` with an actual description in `tools/template.toml`
 - replace `NAME` with the chosen name of the package in:
    - `tools/template.toml`
    - this file below
    - `tools/generate_cli.py` 
    - `.github/workflows/pytest.yml`

# How to install
```bash
pyenv install 3.10 -y
pyenv virtualenv 3.10 NAME
pyenv activate NAME
pyenv local NAME
pip install -e .
```


