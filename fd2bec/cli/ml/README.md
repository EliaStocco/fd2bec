# MACE-POLAR

This workflow computes effective charges for an isolated structure from finite
differences of the total molecular dipole predicted by MACE-POLAR.

Install the optional model dependencies:

```bash
python -m pip install "mace-torch>=0.3.16"
python -m pip install \
  "git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0"
```

MACE-POLAR checkpoints use the Academic Software License (ASL). Review and
accept its terms before downloading or using a checkpoint.

Run the finite-displacement calculation and the existing fd2bec fit:

```bash
mace_polar_dPdR -i molecule.xyz -m polar-1-m -o dataset.extxyz
dPdR2bec -i dataset.extxyz -o results
```

`polar-1-m` is downloaded and cached by MACE on first use. A local checkpoint
path or direct model URL can be supplied with `--model` instead. Charge and spin
multiplicity are read from the input structure's `charge` and `spin` metadata,
defaulting to a neutral singlet; they can be overridden with `--charge` and
`--spin`.

The command deliberately rejects periodic inputs. MACE-POLAR's total dipole is
not well-defined with periodic boundary conditions, so it cannot be used for a
periodic dP/dR calculation.
