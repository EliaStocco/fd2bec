# MACE-POLAR

## Oxidation-number arrays

Some ML models require nominal oxidation numbers as a per-atom extxyz array.
Create a JSON mapping such as:

```json
{"Ba": 2, "Ti": 4, "O": -2}
```

Then add the default `Qs` array to every snapshot:

```bash
add_oxidation_numbers -i structures.extxyz -c oxidation-numbers.json \
  -o charged-structures.extxyz
```

Select another array name with `--name`. Charge neutrality is checked for every
snapshot; use `--allow-non-neutral` only for an intentionally charged system.

## Dipole and polarization workflows

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

The atomic ML adapter uses the same Cartesian displacement implementation as
`generate_displacements`. Run the finite-displacement calculation and the
existing fd2bec fit with:

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

For symmetry and integration testing of the piezoelectric workflow,
`mace_polar_dPdS` generates 13 reference/signed symmetric strain modes,
evaluates the MACE-POLAR cell dipole, and divides it by the cell volume. It
keeps its dedicated symmetric-strain construction because the proper-tensor
regression is sensitive to rotational components of a general cell change:

```bash
mace_polar_dPdS -i periodic.extxyz -m polar-1-m -o dipole-cells.extxyz
dPdS2piezo -i dipole-cells.extxyz -r periodic.extxyz -o piezoelectric
```

`dPdS2piezo` can also consume an ML dataset containing only `REF_dipole`; it
automatically divides each dipole by that snapshot's cell volume.

This is explicitly a polarization proxy, not a Berry-phase polarization.
MACE-POLAR remains useful here as a regression/integration adapter, while
first-principles periodic predictions should use FHI-aims or Quantum ESPRESSO.
See `../dPdS/README.md` for conventions and limitations.
