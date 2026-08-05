# Born effective charges from dP/dR

Generate the finite-displacement structures with:

```bash
generate_displacements -i reference.extxyz --what bec \
  -d displacements.txt -o displaced-structures.extxyz
```

The default uses symmetry. Add `--no-symmetry`, or use `--number N --seed S`
for random displacements.

Use `build_dataset4dPdR.py` for periodic structures and polarization data. Use
`build_dataset4dPdR_nonperiodic.py` for isolated structures (`pbc=False`) and
total-dipole data (for example `-f aims_dipole`). Both produce an extxyz file
that can be passed to `dPdR2bec.py`:

```bash
dPdR2bec -i dataset.extxyz -o bec
bec2charges -i bec/bec.txt -o bec/charges.txt
```

For calculator setup, see `../aims/README.md`, `../qe/README.md`, or the
MACE-POLAR molecular adapter in `../ml/README.md`.
