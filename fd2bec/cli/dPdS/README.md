# Proper and improper piezoelectric tensors

Both tensors can be obtained from one shared set of cell-displaced structures.
Without symmetry reduction, this contains the reference plus positive and
negative versions of six independent lower-triangular cell components.

```bash
generate_displacements -i reference.extxyz --what piezo -a 0.005 \
  --no-symmetry -o displaced-cells.extxyz
```

Evaluate the Cartesian polarization of every frame and save it in the
`REF_polarization` info field. Polarization values must use either e/Å² (the
default) or C/m². Alternatively, store the total dipole in `REF_dipole` in
eÅ; `dPdS2piezo` automatically converts it to e/Å² using the volume of each
snapshot. A separate N×3 polarization text file is also supported.

Dipole inputs are currently supported only in eÅ. Debye, C·m, and atomic-unit
dipoles must be converted to eÅ before running `dPdS2piezo`.

```bash
dPdS2piezo -i polarized-cells.extxyz -r reference.extxyz \
  --polarization-unit 'C/m^2' -o piezoelectric

# Or supply the polarization separately:
dPdS2piezo -i displaced-cells.extxyz -r reference.extxyz \
  -p polarization.txt -o piezoelectric

# Explicitly select a custom dipole field:
dPdS2piezo -i dipole-cells.extxyz -r reference.extxyz \
  --quantity dipole --dipole-keyword my_dipole -o piezoelectric
```

In automatic mode an existing polarization is preferred over a dipole. Use
`--quantity polarization` or `--quantity dipole` when both fields exist and
you need to select one explicitly. When `--polarization-unit C/m^2` is selected,
dipole-derived e/Å² values are converted to C/m² before fitting.

The command writes `improper-piezoelectric.txt` and
`proper-piezoelectric.txt` as 3×6 matrices. It first fits the improper tensor,
then obtains the proper tensor using Vanderbilt's geometric correction. A
second, direct fit using the symmetry modes of `ProperPiezoelectricTensor` is
written to `proper-piezoelectric-direct.txt`. All three matrices are also
printed. Disable crystal-symmetry constraints in the direct fit with
`--no-symmetry`.

The underlying fit is formulated for the total dipole and the lattice vectors,
similarly to the `dPdR2bec` linear system. Six symmetric deformation responses
are fitted. The other three deformation-gradient directions are rigid
rotations, whose exact vector response is imposed as `δμ = ω μ`. The command
also writes `dipole-strain-derivative.txt` (3×6) and
`dipole-lattice-derivative.txt` (3×9). With symmetry enabled, the linear system
is reduced to the symmetry-allowed `ProperPiezoelectricTensor` modes plus the
three reference-dipole components.

The Voigt order is
`xx, yy, zz, yz, xz, xy`, with engineering shear
`(εxx, εyy, εzz, 2εyz, 2εxz, 2εxy)`. Since strain is dimensionless, the
tensor has the same units as the supplied polarization.

The factor of two belongs to the off-diagonal **strain vector**, not to the
stored piezoelectric columns. Thus

```text
delta_P = e_3x6 @ [εxx, εyy, εzz, 2εyz, 2εxz, 2εxy]
```

is equivalent to the full contraction `delta_P_i = e_ijk ε_jk` with
`e_ijk = e_ikj`. For stress, use the Voigt vector without factors of two:

```text
sigma_V = [σxx, σyy, σzz, σyz, σxz, σxy]
```

With the electric-enthalpy convention `f = f0 - E_i P_i`, the converse
fixed-strain response is

```text
delta_sigma_jk = -e_ijk E_i
delta_sigma_V  = -e_3x6.T @ E
```

Therefore a 3×3×3 array is not needed for this contraction: the saved 3×6
matrix is sufficient. Convert to the full array only when another API explicitly
requires Cartesian indices. If a code defines stress or electric enthalpy with
the opposite sign, the displayed minus sign changes accordingly.

The Vanderbilt and direct proper tensors are compared after every fit. Their
maximum absolute difference and pass/fail status are printed and stored in
`fit.json`; change the default absolute tolerance with `--agreement-tolerance`.

The command also prints the symmetry-allowed proper-tensor pattern as a 3×6
matrix. Zeros are forbidden components; `a`, `b`, `c`, ... label independent
parameters, and repeated or signed letters show symmetry relations. The same
machine-readable pattern is stored as `symmetry_pattern` in `fit.json`.

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

Generate displacements and individual `geometry.in` files:

```bash
prepare_aims -i reference.extxyz --what piezo -a 0.005 \
  -o piezoelectric-geometries
```

Run FHI-aims with identical k-point and Berry-phase polarization settings for
all geometries. If the outputs are named with an `n=<index>` field, build the
dataset and evaluate both tensors with:

```bash
build_dataset4dPdS_aims -i aims-results --pattern 'aims.n=*.out' \
  -o aims-piezoelectric.extxyz
dPdS2piezo -i aims-piezoelectric.extxyz -r geometry.in -o piezoelectric
```

The dataset builder converts the FHI-aims polarization from C/m² to e/Å².

## Quantum ESPRESSO

Quantum ESPRESSO computes the Berry-phase polarization one lattice direction
at a time. Prepare all displaced structures and directional inputs with:

```bash
prepare_qe -i reference.extxyz -t template/scf.in --what piezo
export QE="srun /path/to/pw.x"
source sourceme.sh
```

After extracting the three polarization components into an extxyz dataset:

```bash
dPdS2piezo -i qe-piezoelectric.extxyz -r reference.extxyz -o piezoelectric
```

See [`../qe/README.md`](../qe/README.md) for the generated layout.

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
