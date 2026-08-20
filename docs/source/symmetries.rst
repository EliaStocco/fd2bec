Symmetries in molecules and solids
==================================

Molecules have point-group symmetries, whereas periodic solids have space-
group symmetries. The operations are discovered by different libraries and
may initially use different coordinate systems. In
:class:`fd2bec.atomic.AtomicStructure`, both are converted to the same affine
action before they are applied to atoms or tensors.

The common affine operation
---------------------------

fd2bec stores positions as row vectors. Every geometric symmetry operation
is represented by a pair :math:`(R_g,\mathbf{t}_g)` acting as

.. math::

   \mathbf{x}' = \mathbf{x}R_g^T + \mathbf{t}_g.

For all operations at once,
:meth:`fd2bec.atomic.AtomicStructure.get_symmetry_operations` returns arrays
with shapes

.. math::

   R: (N_{\mathrm{op}},3,3),
   \qquad
   T: (N_{\mathrm{op}},3).

The public interface is therefore the same after symmetry discovery:

.. list-table::
   :header-rows: 1
   :widths: 22 25 25 28

   * - Structure
     - Symmetry source
     - Native coordinates
     - fd2bec result
   * - Periodic solid
     - ``spglib`` space group
     - Fractional ``(R, t)``
     - Fractional or Cartesian affine operations
   * - Molecule
     - ``pymatgen`` point group
     - Cartesian ``(R, t)``
     - Cartesian affine operations

The identity, proper rotations, improper rotations, reflections, inversions,
and nonsymmorphic operations all fit this representation. Their physical
meaning differs, but their downstream matrix action does not.

Periodic solids: space-group operations
---------------------------------------

For a periodic structure, ``AtomicStructure`` passes

.. math::

   (A, \{\mathbf{s}_i\}, \{Z_i\})

to :func:`spglib.get_symmetry_dataset`. ``spglib`` returns fractional
rotations :math:`W_g` and translations :math:`\mathbf{w}_g`. They act on a
fractional row vector as

.. math::

   \mathbf{s}'
   = \mathbf{s}W_g^T + \mathbf{w}_g
   \pmod{\mathbb{Z}^3}.

The integer lattice-vector ambiguity is essential: positions separated by an
integer fractional vector are the same site in the infinite crystal. This is
why atom matching wraps fractional differences into the nearest periodic
image.

Calling ``get_symmetry_operations(basis="fractional")`` returns these native
operations directly. Fractional operations are supported only for periodic
structures.

For ``basis="cartesian"``, the :class:`fd2bec.tensor.Rotation` and
:class:`fd2bec.tensor.Translation` tensor definitions convert the operations
using the cell. This preserves the same geometric action in Cartesian
coordinates:

.. math::

   \mathbf{r}' = \mathbf{r}R_{g,\mathrm{cart}}^T
                 + \mathbf{t}_{g,\mathrm{cart}}.

Keeping the basis explicit avoids treating a fractional crystallographic
matrix as though it were already a Cartesian rotation, which would be wrong
for a general non-orthogonal cell.

Molecules: point-group operations
---------------------------------

A molecule has no periodic cell. ``AtomicStructure`` constructs a
:class:`pymatgen.core.Molecule` and uses
:class:`pymatgen.symmetry.analyzer.PointGroupAnalyzer` with ``symprec`` as its
geometric tolerances. The analyzer supplies Cartesian rotations and
translations.

Point-group operations are naturally described relative to a molecular
origin. fd2bec places them back in the coordinate frame of the stored
positions. Let

.. math::

   \mathbf{O} = \frac{1}{N}\sum_i \mathbf{r}_i

be the arithmetic mean of the atomic positions. If ``pymatgen`` supplies
:math:`(R_g,\mathbf{t}_g)`, fd2bec uses the effective translation

.. math::

   \mathbf{t}_{g,\mathrm{eff}}
   = \mathbf{t}_g + \mathbf{O} - \mathbf{O}R_g^T.

The resulting operation

.. math::

   \mathbf{r}'
   = \mathbf{r}R_g^T + \mathbf{t}_{g,\mathrm{eff}}

acts directly on the original, uncentered Cartesian coordinates. Thus a
rotation about the molecular center is represented by the same affine pair
used for a rotation-plus-translation in a crystal.

Symmetry includes atom permutations
-----------------------------------

Applying a geometric operation preserves the structure but does not usually
preserve every atom index. A rotation can move one oxygen site onto another
equivalent oxygen site. For each operation, fd2bec therefore:

1. transforms every stored position with :math:`(R_g,\mathbf{t}_g)`;
2. creates a temporary structure with those positions;
3. uses the species-resolved assignment from the previous chapter;
4. obtains the corresponding atom permutation.

With the mapping convention

.. math::

   m_g(i) = \text{original site corresponding to transformed atom }i,

the code constructs a permutation matrix with

.. math::

   (\Pi_g)_{i,m_g(i)} = 1.

This step is common to molecules and solids. Periodic matching uses wrapped
fractional coordinates; molecular matching uses Cartesian coordinates. After
the mapping is known, both kinds of structure provide the same permutation
object to the tensor machinery.

From geometric operations to tensor operations
-----------------------------------------------

:meth:`fd2bec.atomic.AtomicStructure.get_tensor_symmetry_operations` lifts
the three-dimensional geometric operation to the full space of a selected
tensor. A tensor definition describes its explicit axes. For each axis the
current implementation supplies:

* :math:`\Pi_g` for an atomic axis;
* :math:`R_g^T` for a Cartesian axis.

The matrices are combined in storage order with a Kronecker product by
:meth:`fd2bec.tensor.Tensor.full_operator`. For example, an atomic vector has
one atomic axis and one Cartesian axis, so its flattened linear operator is

.. math::

   G_g = \Pi_g \otimes R_g^T.

A global vector has no atomic axis and receives only the Cartesian operation.
A tensor with several Cartesian axes receives one rotation factor for each
axis. The result always acts on the fully flattened explicit tensor shape:

.. math::

   \mathbf{x}'_{\mathrm{flat}}
   = G_g\mathbf{x}_{\mathrm{flat}} + \mathbf{q}_g.

Here :math:`\mathbf{q}_g` is zero for an ordinary linear tensor.

Affine tensor quantities
~~~~~~~~~~~~~~~~~~~~~~~~

Positions and other tensor definitions with an ``affine`` Cartesian axis may
also receive a translation. fd2bec first broadcasts the geometric
translation over the other tensor axes. It then corrects lattice-image
translations so that the supplied affine reference value is fixed exactly by
each symmetry operation.

When a purely matrix representation is needed, the affine operation is
embedded in homogeneous coordinates:

.. math::

   \begin{pmatrix}\mathbf{x}'\\1\end{pmatrix}
   =
   \begin{pmatrix}
      G_g & \mathbf{q}_g\\
      0   & 1
   \end{pmatrix}
   \begin{pmatrix}\mathbf{x}\\1\end{pmatrix}.

This is the representation averaged by ``get_totally_symmetric_projection``
for affine tensors.

Validation of the unified representation
-----------------------------------------

The private ``_test_symmetry`` method checks the construction operation by
operation. It applies every returned pair to the structure, verifies
structural equality, checks the space-group number for a solid, reconstructs
the atom mapping, and confirms that every mapped positional difference is
within tolerance.

Tests exercise the two discovery paths separately:

* ``tests/test_symmetry.py`` and ``tests/test_spglib.py`` check periodic
  space-group operations;
* ``tests/test_symmetry_molecules.py`` checks point-group operations for
  molecular examples;
* ``tests/test_tensor_symmetry_operations.py`` checks the lifted tensor
  operators and affine translations.

The complete common path is

.. math::

   \boxed{
   \begin{array}{c}
   \text{spglib space group}\\[-0.1cm]
   \text{or}\\[-0.1cm]
   \text{pymatgen point group}
   \end{array}
   \longrightarrow (R_g,\mathbf{t}_g)
   \longrightarrow (\Pi_g,R_g)
   \longrightarrow (G_g,\mathbf{q}_g)
   }.

The next chapter explains how ``get_symmetrizer`` averages these common
tensor operations and extracts the symmetry-allowed degrees of freedom.
