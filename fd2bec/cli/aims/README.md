# FHI-aims preparation

`get_basis_functions_fhi_aims` creates a `species.in` file by collecting the unique elements
from the first input structure and concatenating their standard FHI-aims files:

```bash
get_basis_functions_fhi_aims -i geometry.in -b light -f /path/to/fhi-aims
```

If `-f/--folder` is omitted, the command reads the FHI-aims location from
`$AIMS_PATH` (or the variable selected with `--variable`). Use `-o` to choose
the output filename.

`prepare_aims` uses the same lookup automatically when `control.in` has no
active `species` blocks. It creates `species.<basis>.in` and appends it to
`control.in`; existing species blocks are not duplicated. Use `--basis`,
`--aims-folder`, or `--aims-variable` to configure this lookup.

`prepare_aims` requires a `control.in` in the current directory, accepts one periodic reference structure, and runs the shared
displacement generator and geometry exporter:

```bash
prepare_aims -i reference.extxyz --what bec
prepare_aims -i reference.extxyz --what piezo
```

Use `--no-symmetry` for all signed basis displacements or `--number N --seed S`
for random displacements. `--k-density D` determines the k-grid, replacing
any existing `k_grid` or `k_grid_density` setting. Use `--k-grid NX NY NZ` to
specify the SCF k-grid explicitly instead; it takes precedence over
`--k-density`. The matching three
`output polarization` lines are also written to `control.in`. Use
`--k_density_polarization D` to choose the absolute reciprocal-space density
along each polarization direction, or `--k-grid-polarization NX NY NZ` to set
that grid explicitly. The explicit grid takes precedence over the polarization
density, and each of its dimensions must exceed the corresponding SCF-grid
dimension. The generated polarization meshes always contain more k-points
than the SCF mesh.

The command writes a multi-frame extxyz file, a text displacement table,
individual AIMS geometries, a log, and `sourceme.sh`. Set `AIMS` in the
submission script and source the helper:

```bash
export AIMS=/path/to/aims
source sourceme.sh
```

Use identical k-points and polarization settings for every displaced geometry.
The preparation command updates `control.in` and prints the selected settings.

For more than one generated geometry, `--use-csc` (or `--csc`) makes the first
calculation write converged ELSI density-matrix restart files and makes later
calculations read them. The flag is automatically disabled for a single
geometry. When CSC is enabled, set `USE_CSC=false` in the submission script
before sourcing `sourceme.sh` to override the default without rerunning
`prepare_aims`. The generated `sourceme.sh` also contains `DELETE_CSC=true`;
set it to `false` there if the final `.csc` files should be retained.

## Post-processing

`post_process_aims` is intentionally a Born-charge-only convenience wrapper:

```bash
post_process_aims -i reference.extxyz --results results -o bec
```

For piezoelectric calculations use:

```bash
build_dataset4dPdS_aims -i results --pattern 'aims.n=*.out' \
  -o aims-piezoelectric.extxyz
dPdS2piezo -i aims-piezoelectric.extxyz -r reference.extxyz \
  -o piezoelectric
```

Use `prepare_aims --what piezo` to prepare FHI-aims piezoelectric geometries.
