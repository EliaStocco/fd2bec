# Proper and improper piezoelectric tensors

Both tensors can be obtained from one set of 13 strained structures: an
unstrained reference and positive/negative versions of the six symmetric
strain modes.

```bash
generate_strained_structures -i reference.extxyz -a 0.005 \
  -o strained-structures.extxyz
```

Evaluate the Cartesian polarization of every frame and save it in the
`REF_polarization` info field. Polarization values must use either e/Å² (the
default) or C/m². Alternatively, write the values as an N×3 text file.

```bash
dPdS2piezo -i polarized-strained-structures.extxyz -r reference.extxyz \
  --polarization-unit 'C/m^2' -o piezoelectric

# Or supply the polarization separately:
dPdS2piezo -i strained-structures.extxyz -r reference.extxyz \
  -p polarization.txt -o piezoelectric
```

The command writes `improper-piezoelectric.txt` and
`proper-piezoelectric.txt` as 3×6 matrices. The Voigt order is
`xx, yy, zz, yz, xz, xy`, with engineering shear
`(εxx, εyy, εzz, 2εyz, 2εxz, 2εxy)`. Since strain is dimensionless, the
tensor has the same units as the supplied polarization.

The improper tensor is fitted directly from the Cartesian polarization:

```text
c_ijk = dP_i / dε_jk
```

The proper tensor is obtained from the same fit and reference polarization:

```text
c~_ijk = c_ijk + δ_jk P_i - ½(δ_ij P_k + δ_ik P_j)
```

The last two terms are symmetrized because this workflow applies symmetric
strain rather than a general deformation gradient. Before fitting, polarization
branches are aligned in reduced coordinates using the polarization quantum of
each strained cell. The `--polarization-unit` option supplies the necessary SI
conversion while preserving that unit in the output. Use `--no-unwrap` only if
the input polarizations have already been aligned.

Keeping fractional coordinates fixed gives the clamped-ion response. If each
strained structure is internally relaxed before its polarization is evaluated,
the same post-processing gives the relaxed-ion response.

## FHI-aims

Generate individual `geometry.in` files:

```bash
aims_geometries4dPdS -i geometry.in -a 0.005 -o piezoelectric-geometries
```

Run FHI-aims with identical k-point and Berry-phase polarization settings for
all 13 geometries. If the outputs are named with an `n=<index>` field, build the
dataset and evaluate both tensors with:

```bash
build_dataset4dPdS_aims -i aims-results --pattern 'aims.n=*.out' \
  -o aims-piezoelectric.extxyz
dPdS2piezo -i aims-piezoelectric.extxyz -r geometry.in -o piezoelectric
```

The dataset builder converts the FHI-aims polarization from C/m² to e/Å².

## MACE-POLAR

```bash
mace_polar_dPdS -i periodic.extxyz -m polar-1-m \
  -o mace-polar-piezoelectric.extxyz
dPdS2piezo -i mace-polar-piezoelectric.extxyz -r periodic.extxyz \
  -o piezoelectric
```

MACE-POLAR is a molecular model and reports a total dipole, not a periodic
Berry-phase polarization. This adapter uses dipole divided by cell volume as a
polarization proxy. It is useful for symmetry and integration tests, including
the centrosymmetric BiFeO₃ regression, but it should not be interpreted as a
general first-principles periodic piezoelectric prediction.
