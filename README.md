# Finite Differences to Born Effective Charges (fd2bec)

A Python package for finite-difference Born effective charges and proper and
improper piezoelectric tensors.

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

# FHI-aims workflows

Starting from a periodic structure, generate either atomic displacements for
Born charges or cell displacements for piezoelectric tensors:

```bash
prepare_aims -i start.extxyz --what bec
# or
prepare_aims -i start.extxyz --what piezo
```

Provide `control.in`, set `AIMS` in the submission script, and source the
generated `sourceme.sh`. `post_process_aims` is a BEC-only convenience wrapper.
For piezoelectric results use `build_dataset4dPdS_aims` followed by
`dPdS2piezo`. See [`fd2bec/cli/aims/README.md`](fd2bec/cli/aims/README.md).

# Quantum ESPRESSO workflows

Provide one reference structure and an SCF template containing `! FD2BEC`:

```bash
prepare_qe -i start.extxyz -t template/scf.in --what bec
# or
prepare_qe -i start.extxyz -t template/scf.in --what piezo
```

The command generates the displaced structures, QE geometry cards, SCF/NSCF
templates, and `sourceme.sh`. See
[`fd2bec/cli/qe/README.md`](fd2bec/cli/qe/README.md).

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

Generate cell-displaced structures, evaluate their dipoles, and use the
same dataset to fit both tensors:

```bash
generate_displacements -i reference.extxyz --what piezo \
  -o displaced-cells.extxyz
dPdS2piezo -i dipole-cells.extxyz -r reference.extxyz -o piezoelectric
```

See `fd2bec/cli/dPdS/README.md` for the strain convention, dipole input,
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
