# FHI-aims preparation

`prepare_aims` accepts one periodic reference structure and runs the shared
displacement generator and geometry exporter:

```bash
prepare_aims -i reference.extxyz --what bec
prepare_aims -i reference.extxyz --what piezo
```

Use `--no-symmetry` for all signed basis displacements or `--number N --seed S`
for random displacements. The command writes a multi-frame extxyz file, a text
displacement table, individual AIMS geometries, a log, and `sourceme.sh`.
Provide a `control.in`, set `AIMS` in the submission script, and source the
helper:

```bash
export AIMS=/path/to/aims
source sourceme.sh
```

Use identical k-points and polarization settings for every displaced geometry.
The preparation command prints suggested settings.

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

`aims_geometries4dPdS` is retained for compatibility; new workflows should use
`prepare_aims --what piezo`.
