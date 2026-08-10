# Proper and improper piezoelectric tensors

Both tensors can be obtained from one shared set of cell-displaced structures.
Without symmetry reduction, this contains the reference plus positive and
negative versions of six independent lower-triangular cell components.

```bash
generate_displacements -i reference.extxyz --what piezo -a 0.005 \
  --no-symmetry -o displaced-cells.extxyz
```

Store the total dipole of every frame in the `REF_dipole` info field. Dipoles
must be in e·Å; Debye, C·m, and atomic-unit dipoles must be converted before
running `dPdS2piezo`. The command converts each dipole to polarization using
that frame's cell volume.

```bash
dPdS2piezo -i dipole-cells.extxyz -r reference.extxyz -o piezoelectric

# Select a custom dipole field, still in e·Å:
dPdS2piezo -i dipole-cells.extxyz -r reference.extxyz \
  --dipole-keyword my_dipole -o piezoelectric
```

Periodic dipoles can be multivalued. Branch alignment is enabled by default,
using each snapshot's cell-dependent polarization quantum after conversion to
polarization. Use `--no-unwrap` only when the dipoles are already on a
consistent branch. Do not preprocess strained datasets with `build_dataset4dPdR`:
that builder requires identical cells and volumes and is intended for atomic,
not lattice, displacements.

The command writes `improper-piezoelectric.txt` and
`full-piezoelectric.txt` as 3×6 matrices. It first fits the improper tensor,
then obtains the full tensor from the dipole/lattice linear system using
Vanderbilt's geometric correction. A second fit using the symmetry modes of
`ProperPiezoelectricTensor` is called the clamped tensor and is written to
`clamped-piezoelectric.txt`. All three matrices are also printed. Disable
crystal-symmetry constraints in the clamped fit with `--no-symmetry`.

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
tensor is reported in e/Å² because input dipoles are in e·Å. The lattice-basis
tensor contains two lattice-vector factors and one inverse lattice-vector
factor, so its printed unit is e/Å.
Numerical 3×6 tensors are printed as aligned tables with `P_x`, `P_y`, and
`P_z` row labels and explicit Voigt-column headers. The lattice-basis tensor is
printed as three labeled 3×3 slices. Values below the display tolerance are
shown as `0.000000`, avoiding distracting negative zeros without changing the
saved numerical data.

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

For the default `--clamped` workflow, every frame must retain the reference
fractional coordinates. The full and clamped tensors are then compared; their
maximum absolute difference and pass/fail status are printed and stored in
`fit.json`. Change the default absolute tolerance with
`--agreement-tolerance`. For internally relaxed structures, pass
`--no-clamped`; fractional coordinates are then expected to change and the
tensor-agreement check is skipped.

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

The improper tensor is fitted directly from the dipole-derived polarization:

```text
c_ijk = dP_i / dε_jk
```

The full tensor is obtained from the same fit and reference polarization:

```text
c~_ijk = c_ijk + δ_jk P_i - ½(δ_ij P_k + δ_ik P_j)
```

The last two terms are symmetrized because this workflow applies symmetric
strain rather than a general deformation gradient. Before fitting, dipole
branches are aligned in reduced coordinates using the polarization quantum of
each strained cell. Use `--no-unwrap` only if the input dipoles have already
been aligned.

Keeping fractional coordinates fixed gives the clamped-ion response. If each
strained structure is internally relaxed before its polarization is evaluated,
the same post-processing with `--no-clamped` gives the relaxed-ion response.

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

The dataset builder converts the FHI-aims polarization from C/m² to a total
dipole in e·Å, which is stored in `REF_dipole` for `dPdS2piezo`.

## Quantum ESPRESSO

Quantum ESPRESSO computes the Berry-phase polarization one lattice direction
at a time. Prepare all displaced structures and directional inputs with:

```bash
prepare_qe -i reference.extxyz -t template/scf.in --what piezo
export QE="srun /path/to/pw.x"
source sourceme.sh
```

After extracting total dipoles in e·Å into an extxyz dataset:

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
