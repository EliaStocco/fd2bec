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

Berry-phase polarizations are aligned by default in reduced coordinates using
the polarization quantum of each snapshot's changing cell. A dipole from a
periodic Berry-phase calculation is also multivalued: multiplying polarization
by the cell volume does not remove its branch ambiguity. Dipole inputs are
therefore divided by each snapshot's volume and branch-aligned by default too.
Use `--no-unwrap` only when the input vectors have already been placed on a
consistent branch. The older `--unwrap-dipoles` option is retained for command
compatibility but is no longer necessary. Do not preprocess strained datasets
with `build_dataset4dPdR`: that builder requires identical cells and volumes
and is therefore intended for atomic, not lattice, displacements.

The command writes `improper-piezoelectric.txt` and
`proper-piezoelectric.txt` as 3×6 matrices. It first fits the improper tensor,
then obtains the proper tensor using Vanderbilt's geometric correction. A
second, direct fit using the symmetry modes of `ProperPiezoelectricTensor` is
written to `proper-piezoelectric-direct.txt`. All three matrices are also
printed. Disable crystal-symmetry constraints in the direct fit with
`--no-symmetry`.

For comparison, the command additionally prints the Vanderbilt proper tensor
in the reference lattice (fractional) basis using the tensor class's basis
transformation. It is shown as the full 3×3×3 tensor. The usual 3×6
engineering-Voigt contraction is reserved for an orthonormal Cartesian basis
and is therefore not applied to these lattice-basis components.

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
tensor has the same units as the supplied polarization. These units are shown
in every numerical Cartesian tensor and named-coefficient heading:
`e/Angstrom^2` by default or `C/m^2` when requested. The lattice-basis tensor
contains two lattice-vector factors and one inverse lattice-vector factor;
because cells are represented in Angstrom, its printed unit is `e/Angstrom`
or `C*Angstrom/m^2`, respectively.

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

The command first prints a compact symbolic 3×6 matrix in which `a`, `b`, ...
denote independent parameters. It then prints every symmetry-allowed
`ProperPiezoelectricTensor` mode as a separate numeric 3×6 matrix. A
rank-revealing decomposition selects well-conditioned independent tensor
components. Each canonical mode sets its labeled component to 1 and the other
selected independent components to 0, so the display is reproducible rather
than dependent on arbitrary eigensolver vectors. The representations and
anchor indices are stored as `symmetry_symbolic_matrix`, `symmetry_modes`, and
`symmetry_mode_independent_components` in `fit.json`.
The modes are expressed in the input Cartesian frame. For a rhombohedral
structure represented in pseudocubic Cartesian axes, the threefold axis is not
a Cartesian axis, so individual modes can contain several nonzero Cartesian
components. A textbook trigonal representation requires first rotating the
structure and tensor to symmetry-adapted axes. Pass `--conventional-axes` to
rotate the reported and saved piezoelectric tensors, reference polarization,
symbolic matrix, and numeric modes into spglib's conventional Cartesian axes.
The fitted data are not modified; the coordinate rotation is applied after the
fit and is printed and stored in `fit.json`.
The detected international space-group number and symbol, point group, number
of symmetry operations, and number of allowed proper-piezoelectric parameters
are printed before the fit and recorded in `fit.json`.
The command also prints the unstrained reference cell vectors, lattice
parameters `(a, b, c, alpha, beta, gamma)`, and wrapped fractional coordinates
with atom indices and chemical symbols. These describe the input reference
frame, even when `--conventional-axes` is used for the reported tensors.

The command chooses one reproducible set of independent matrix entries from
the canonical symmetry modes. Their names are generated directly from their
3×6 positions as `eij`, where `i` is the polarization row and `j` is the Voigt
column. Each selected name and fitted value is printed after the direct-fit
matrix and stored in `fit.json`. The selection refers to the displayed frame:
input Cartesian axes by default or conventional axes with
`--conventional-axes`; it does not depend on an external tabulation.

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
