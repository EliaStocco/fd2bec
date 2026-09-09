# Quantum ESPRESSO preparation

Provide one periodic reference structure and an SCF template containing exactly
one `! FD2BEC` marker inside `&CONTROL`:

```bash
prepare_qe -i reference.extxyz -t template/scf.in --what bec
prepare_qe -i reference.extxyz -t template/scf.in --what piezo
```

The template must use `K_POINTS automatic`. For each direction `gdir`, the
generated NSCF input uses Berry phase polarization and
`nppstr = 10 * k_grid[gdir]`; change the multiplier with `--nppstr-factor`.

The output contains:

```text
qe-calculations/
├── displaced-structures.extxyz
├── displacements.txt
├── geometries/geometry.n=<index>.in
├── templates/scf.in
├── templates/nscf.g=<1,2,3>.in
├── work/                         # created while running
└── results/geometry.n=<index>/   # SCF and three NSCF outputs
```

In the submission script, define the complete `pw.x` launch command and source
the generated helper:

```bash
export QE="srun /path/to/pw.x"
source qe-calculations/sourceme.sh
```

`sourceme.sh` is stored inside the output folder and resolves every file
relative to its own location. The complete output folder can therefore be
copied or moved to a cluster without editing paths.

The helper appends each cell/fractional-coordinate geometry to the SCF and NSCF
templates, runs SCF first, then one NSCF calculation for each polarization
direction, and skips non-empty output files that already exist.
