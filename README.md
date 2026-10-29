```text
┏━╸╺┳┓┏━┓┏┓ ┏━╸┏━╸  |  Finite Differences     
┣╸  ┃┃┏━┛┣┻┓┣╸ ┃    |    to                     
╹  ╺┻┛┗━╸┗━┛┗━╸┗━╸  |  Born Effective Charges 
```

<!-- # Finite Differences to Born Effective Charges (fd2bec) -->

**fd2bec** is a Python toolkit for computing Born effective charge tensors and
proper or improper piezoelectric tensors from finite-difference calculations.
It generates symmetry-reduced atomic and cell displacements, organizes the
resulting datasets, and fits response tensors from polarization, dipole, or
force data.

The package supports periodic crystals and isolated molecules, using a unified
symmetry framework based on space groups (`spglib`) and molecular point groups
(`pymatgen`). It provides workflows for FHI-aims and Quantum ESPRESSO, together
with a MACE-POLAR adapter for molecular dipole-based effective-charge
calculations. Structures are exchanged through ASE-compatible formats,
especially extended XYZ.

Beyond the end-to-end workflows, fd2bec includes reusable tools for structure
matching and reordering, symmetry projections and allowed tensor modes,
polarization-branch alignment, strain conventions, tensor conversion, and
post-processing of fitted results. It is not an electronic-structure code
itself: it prepares finite-difference calculations and turns their computed
observables into physically constrained response tensors.

# Installation

fd2bec requires Python 3.9 or newer. Python 3.11 is recommended.

```bash
pyenv install 3.11
pyenv virtualenv 3.11 fd2bec
pyenv activate fd2bec
pyenv local fd2bec
python -m pip install --upgrade pip
python -m pip install -e .
```

To build the mathematical documentation, install the development dependencies
and ensure that a LaTeX installation is available:

```bash
python -m pip install -e ".[dev]"
./tools/build_docs.sh          # creates fd2bec_math.pdf in the root directory
```

See [`docs/README.md`](docs/README.md) for the documentation build details.

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
# or prepare both response calculations without output-name collisions
prepare_aims -i start.extxyz --what both
```

Provide `control.in`, set `AIMS` in the submission script, and source the
generated `sourceme.sh`. `post_process_aims --what both -i start.extxyz` then
fits both response tensors. See
[`fd2bec/cli/aims/README.md`](fd2bec/cli/aims/README.md).

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
python -m pip install -e ".[mace-polar]"
mace_polar_dPdR -i molecule.xyz -m polar-1-m -o dataset.extxyz
dPdR2bec -i dataset.extxyz -o results
```

The Born effective charges are written to `results/bec.txt`. MACE-POLAR's total
dipole is only meaningful for isolated structures, so this workflow rejects
periodic inputs. See `fd2bec/cli/ml/README.md` for model, charge, spin, and
licensing details.

# Alternative Born-charge workflow: dF/dE

Born effective charges can also be obtained from the derivative of atomic
forces with respect to an applied electric field. After calculating forces for
the same structure at several electric fields, collect the results and fit the
tensor:

```bash
build_dataset4dFdE -i field-calculations -o field-forces.extxyz
dFdE2bec -i field-forces.extxyz -o results
```

This route is useful when force data under finite electric fields are available
from the chosen electronic-structure code.

To visualise the directional responses of all Born effective charges in an
extxyz field (normally `REF_BEC`):

```bash
plot_bec -i born-charges.extxyz --all-atoms -o born-charge-components.png
```

This writes a three-column figure with one atom per row. For a displacement
direction `n`, the columns show the signed longitudinal charge `n_i Z*_ij n_j`,
the total induced-polarization magnitude `|Z*_ij n_j|`, and the part transverse
to `n`. Use `--atom INDEX` instead to plot just one atom.

The BaTiO3 generator creates a common-scale grid for each phase from the
`MACE_BEC` tensors in `examples/domains/structures`:

| Cubic | Tetragonal |
| --- | --- |
| ![Cubic BaTiO3 Born effective charge directional responses](docs/source/_static/piezoelectric/bec/batio3-cubic-bec-response-components.png) | ![Tetragonal BaTiO3 Born effective charge directional responses](docs/source/_static/piezoelectric/bec/batio3-tetragonal-bec-response-components.png) |
| Orthorhombic | Rhombohedral |
| ![Orthorhombic BaTiO3 Born effective charge directional responses](docs/source/_static/piezoelectric/bec/batio3-orthorhombic-bec-response-components.png) | ![Rhombohedral BaTiO3 Born effective charge directional responses](docs/source/_static/piezoelectric/bec/batio3-rhombohedral-bec-response-components.png) |

For atom-arrow visualisation in OVITO, export an extxyz file for a chosen
displacement direction. It retains the full tensor and adds `BEC_response`
(the arrow vector), plus longitudinal, magnitude, and transverse scalar fields:

```bash
export_bec_visualization -i born-charges.extxyz -o bec-arrows.extxyz \
  --direction 0 0 1
```

Load `bec-arrows.extxyz` in OVITO and select `BEC_response` in the Vector
Visualization modifier. A cube file is not generated because Born effective
charges are atom-centred tensors, rather than a scalar field on a grid.

Regenerate all checked-in BaTiO3 piezoelectric and Born-charge figures with:

```bash
./tools/generate_batio3_directional_plots.sh
```

Pass an output directory as its first argument, or set `PYTHON` to select the
Python executable used to run the plotting and export commands. The command
transcript is appended to `logs/generate-batio3-directional-plots.log` within
that output directory.

# Computing piezoelectric tensors

Generate cell-displaced structures, evaluate their dipoles, and use the
same dataset to fit both tensors:

```bash
generate_displacements -i reference.extxyz --what piezo \
  -o displaced-cells.extxyz
dPdS2piezo -i dipole-cells.extxyz -r reference.extxyz -o piezoelectric
```

See `fd2bec/cli/dPdS/README.md` for the strain convention, dipole input,
Berry-phase branch handling, and clamped-ion workflow.

To visualise the directional anisotropy, plot the proper tensor written by
that command:

```bash
plot_piezoelectric -i piezoelectric/proper-piezoelectric.txt \
  -o piezoelectric/directional-surface.png --unit 'e/Angstrom^2'

# Write longitudinal, total, and transverse induced-polarization surfaces together.
plot_piezoelectric -i piezoelectric/proper-piezoelectric.txt \
  -o piezoelectric/directional-responses.png --response all
```

The surface represents the signed longitudinal coefficient
`n_i e_ijk n_j n_k` for each unit direction `n`: its radius is the absolute
coefficient and its colour gives the sign.  The input uses the Voigt order
`xx, yy, zz, yz, xz, xy` and the package's engineering-shear convention.
`--response transverse` instead plots the magnitude of the induced
polarization perpendicular to `n`; `--response all` writes longitudinal,
total, and transverse surfaces side by side.

The gallery below uses the clamped-ion BaTiO3 example tensors in
`examples/BaTiO3-piezo-clamped`. All four images use the same radial and colour
scale (±0.03046 e/Angstrom²), so their lobe sizes are comparable. The cubic
phase has no piezoelectric response, as expected for its centrosymmetric
structure.

| Cubic | Tetragonal |
| --- | --- |
| ![Cubic BaTiO3 has a zero directional piezoelectric response](docs/source/_static/piezoelectric/piezo/batio3-cubic.png) | ![Tetragonal BaTiO3 longitudinal piezoelectric response](docs/source/_static/piezoelectric/piezo/batio3-tetragonal.png) |
| Orthorhombic | Rhombohedral |
| ![Orthorhombic BaTiO3 longitudinal piezoelectric response](docs/source/_static/piezoelectric/piezo/batio3-orthorhombic.png) | ![Rhombohedral BaTiO3 longitudinal piezoelectric response](docs/source/_static/piezoelectric/piezo/batio3-rhombohedral.png) |

For tetragonal BaTiO3, the combined plot makes the transverse contribution
explicit: left is signed longitudinal response, middle is total induced
polarization, and right is the magnitude perpendicular to the applied strain.

![Tetragonal BaTiO3 longitudinal, total, and transverse directional piezoelectric responses](docs/source/_static/piezoelectric/piezo/batio3-tetragonal-response-components.png)

For a simple mathematical explanation of the symmetry reduction used
throughout the package, see the [symmetry-mode documentation](docs/source/symmetry_modes.rst).

# Testing

Install the development dependencies, then run:

```bash
pytest --ff --nf -x
```

# For developers

Install the development dependencies. After adding or removing command-line
scripts, regenerate the auto-generated package configuration:

```bash
python -m pip install -e ".[dev]"
./tools/initialize.sh
./tools/fix_code.sh
```

For MACE-POLAR development, install `".[dev,mace-polar]"` instead.
