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

When adding a command, update `pyproject.toml` by running `tools/initialize.sh`.
