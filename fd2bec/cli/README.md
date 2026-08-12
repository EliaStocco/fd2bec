# Command-line workflows

`generate_displacements` is the shared structure generator:

```bash
# Born-charge atomic displacements
generate_displacements -i reference.extxyz --what bec -o displaced.extxyz

# Piezoelectric cell displacements
generate_displacements -i reference.extxyz --what piezo -o displaced.extxyz

# Elastic cell/strain displacements
generate_displacements -i reference.extxyz --what elastic -o elastic-displaced.extxyz

# Force-constant atomic displacements
generate_displacements -i reference.extxyz --what force-constants -o force-displaced.extxyz
```

`tensor_symmetries` accepts `--conventional-axes` to rotate the reported
Cartesian component patterns into spglib's conventional crystallographic frame.
It reports the symmetry-inequivalent components for every tensor; Voigt notation
is added when the definition contains a symmetric strain pair.
Repeated atomic blocks are grouped under one atom-index label.

`space_group` prints the input cell, Cartesian and fractional positions, and
the space-group information detected by spglib. Use `--show-operations` to
print the fractional-coordinate symmetry operations and `--threshold` to
change the symmetry tolerance.

`space_group_dataset` reads every frame of a multi-frame extxyz and writes one
CSV row per structure with atom count, volume, lattice parameters, space-group
symbols, crystal class, Bravais lattice type, centrosymmetry, and operation
count. It also creates a companion PNG with histograms of the lattice
parameters, volume, atom count, and operation count, a space-group frequency
bar plot, and an atoms-versus-operations correlation plot. Use `--plot-output`
to choose a different image path.

By default symmetry-inequivalent signed displacements are selected. Use
`--no-symmetry` for every signed Cartesian basis displacement, or `--number N`
and `--seed` for random displacements. `extxyz2folder` converts a multi-frame
file to one geometry per snapshot, including QE geometry-card output with
`--format espresso-in`.

The higher-level `prepare_aims` and `prepare_qe` commands run these steps
automatically. See the README files in `aims`, `qe`, `dPdR`, `dPdS`, and `ml`
for calculator-specific workflows.

`generate_random_displacements` and `aims_geometries4dPdS` are retained only
for compatibility and emit deprecation warnings. Prefer
`generate_displacements --number N` and `prepare_aims --what piezo`.

`split_extxyz` writes each snapshot from a multi-frame extxyz file to
`structure-<n>/start.extxyz` and creates an editable `run_all.sh` that runs
in the current directory. For large inputs, use `--zip` to stream the
snapshots directly into a compressed archive without creating all directories.
The external runner targets the folder created when the archive is extracted:

```bash
split_extxyz -i dataset.extxyz -o dataset --zip
unzip dataset.zip
./run_all.sh
```

The generated `prepare_aims` example accepts `--k-density`; change `5.0` in
`run_all.sh` to use a different reciprocal-space grid density. Use
`--k_density_polarization` to set the absolute density along each
polarization direction.

When adding a command, update `pyproject.toml` by running `tools/initialize.sh`.
