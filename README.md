# Finite Differences to Born Effective Charges (fd2bec)
A python package to efficiently evaluate Born Effective Charges using Finite Differences.

# How to install
```bash
pyenv install 3.11             # it works with all versions from 3.9 to 3.14
pyenv virtualenv 3.11 fd2bec
pyenv activate fd2bec
pyenv local fd2bec
pip install --upgrade pip
./tools/initialize.sh          # for developers only, harmless anyway otherwise
pip install -e .               # only editable mode is fully tested so far
```

For developers we really recommend using `python>=3.11`.

# Finding command-line scripts

After installation, `fd2bec-help` lists the package's command-line tools by
workflow. Use `-f` to select a workflow and `-d` to show descriptions:

```console
$ fd2bec-help -f aims -d

    Looking for scripts in '.../fd2bec/cli'

    aims:
     - post_process_aims.py: Post process calculations from FHI-aims.
     - prepare_aims.py     : Prepare calculations for FHI-aims.
```

Run a listed script without the `.py` suffix, for example `prepare_aims --help`.
Use `fd2bec-help --help` to see all available filters.

# Computing Born Effective Charges with FHI-aims
Let's suppose that you want to compute the Born Effective Charges for a periodic structure stored in `start.extxyz`. 
Let's also suppose that you already have a `control.in` and a submission script `submit.sh`.
To compute the Born Effective Charges using FHI-aims please follow these instructions:
```bash
# Let's start!
prepare_aims -i start.extxyz
# Read the output message of 'prepare_aims'
# and modify 'submit.sh' and 'control.in' accordingly
sbatch submit.sh # let's wait ...
post_process_aims -i start.extxyz
# The Born Effective Charges will be in 'bec.txt'.
# Nicely done!
```

Please do not save your structure in `geometry.in` because this file will be overwritten!

# Computing effective charges with MACE-POLAR

For an isolated structure, `mace_polar_dPdR` predicts total dipoles for all
positive and negative Cartesian displacements and writes a dataset that can be
used by the existing `dPdR2bec` command:

```bash
pip install -e ".[mace-polar]"
mace_polar_dPdR -i molecule.xyz -m polar-1-m -o dataset.extxyz
dPdR2bec -i dataset.extxyz -o results
```

The Born effective charges are written to `results/bec.txt`. MACE-POLAR's total
dipole is only meaningful for isolated structures, so this workflow rejects
periodic inputs. See `fd2bec/cli/ml/README.md` for model, charge, spin, and
licensing details.

# Computing piezoelectric tensors

Generate one shared set of strained structures, evaluate their polarizations,
and use it to fit both the proper and improper piezoelectric tensors:

```bash
generate_strained_structures -i reference.extxyz -o strained.extxyz
dPdS2piezo -i polarized-strained.extxyz -r reference.extxyz -o piezoelectric
```

See `fd2bec/cli/dPdS/README.md` for the strain convention, polarization input,
Berry-phase branch handling, and clamped-ion versus relaxed-ion workflows.

# Testing
We would recommend running tests using
```bash
pytest --ff --nf -x
``` 

# For developers
Run
```bash
pip install -e .[dev] # or .[dev-mp]
./tools/fix_code.sh
```
