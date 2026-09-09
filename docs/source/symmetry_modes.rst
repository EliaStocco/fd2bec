The mathematics of symmetry projections and modes
=================================================

The method

.. py:method:: fd2bec.atomic.AtomicStructure.get_symmetry_modes(tensor, atol=ATOL)

finds all tensor patterns that are unchanged by the symmetries of an atomic
structure. These patterns are later used to reduce a fitting problem: instead
of fitting every tensor component independently, fd2bec fits only components
that symmetry allows.

The main idea is simple:

1. list the ways in which the structure can look unchanged;
2. write each way as a matrix acting on the tensor components;
3. average those matrices;
4. keep the directions that the average leaves unchanged.

No group theory is needed to follow this argument. A ``symmetry operation``
is simply a move that leaves the structure looking the same, such as a
reflection, a rotation, or an exchange of two identical atoms.

What does “unchanged” mean?
----------------------------

Start with a vector of numbers ``x``. A symmetry operation changes it to a new
vector. For a linear operation, the change can be written as

.. math::

   \mathbf{x}' = G\mathbf{x}.

The vector is unchanged when

.. math::

   G\mathbf{x} = \mathbf{x}.

For example, reflect a two-dimensional vector in the horizontal axis:

.. math::

   \begin{pmatrix}x\\y\end{pmatrix}
   \longmapsto
   \begin{pmatrix}x\\-y\end{pmatrix}.

The unchanged vectors have ``y = 0``. The horizontal direction is allowed;
the vertical direction is not.

An even simpler example is a pair of identical atoms that can be exchanged.
For two numbers ``a`` and ``b``, the exchange is

.. math::

   \begin{pmatrix}a\\b\end{pmatrix}
   \longmapsto
   \begin{pmatrix}b\\a\end{pmatrix}.

An unchanged pair must satisfy ``a = b``. The symmetry has removed one
independent number.

From many operations to one projection
---------------------------------------

Suppose the structure has ``M`` allowed operations, represented by
``G_1, G_2, ..., G_M``. fd2bec averages them:

.. math::

   P = \frac{1}{M}\sum_{g=1}^{M} G_g.

This matrix ``P`` is called a projection because it removes the forbidden
parts of a vector and keeps the allowed parts.

Formally, ``P`` is a projection: it is idempotent, meaning that
``P^2 = P``. The allowed operations form a finite group, so the product of
any two operations is another operation in the same list. Hence

.. math::

   \begin{aligned}
   P^2
   &= \left(\frac{1}{M}\sum_{g=1}^{M}G_g\right)
      \left(\frac{1}{M}\sum_{h=1}^{M}G_h\right) \\
   &= \frac{1}{M^2}\sum_{h=1}^{M}\sum_{g=1}^{M}G_gG_h \\
   &= \frac{1}{M^2}\sum_{h=1}^{M}\sum_{\ell=1}^{M}G_\ell
    = \frac{1}{M}\sum_{\ell=1}^{M}G_\ell
    = P.
   \end{aligned}

In the third line, for each fixed ``G_h``, right multiplication by ``G_h``
only permutes the group elements, so ``\{G_gG_h\}`` is the same set as
``\{G_\ell\}``. Thus applying the average twice has exactly the same effect
as applying it once.

For the reflection example,

.. math::

   P = \frac{1}{2}
   \left[
   \begin{pmatrix}1&0\\0&1\end{pmatrix}
   +
   \begin{pmatrix}1&0\\0&-1\end{pmatrix}
   \right]
   =
   \begin{pmatrix}1&0\\0&0\end{pmatrix}.

Therefore

.. math::

   P\begin{pmatrix}x\\y\end{pmatrix}
   =
   \begin{pmatrix}x\\0\end{pmatrix}.

The average keeps the horizontal component and removes the vertical one.

Why does averaging work?
~~~~~~~~~~~~~~~~~~~~~~~~

Applying one more allowed operation only rearranges the terms in the average.
For every operation ``G_k``,

.. math::

   G_k P
   = \frac{1}{M}\sum_{g=1}^{M}G_kG_g
   = \frac{1}{M}\sum_{g=1}^{M}G_g
   = P.

Consequently, every vector produced by ``P`` is unchanged by every symmetry.

Why not solve each constraint separately?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

That is a valid and equivalent approach. Imposing every symmetry condition
``G_g x = x`` separately gives the stacked linear system

.. math::

   A\mathbf{x}=0,
   \qquad
   A=
   \begin{pmatrix}
   G_1-I \\
   G_2-I \\
   \vdots \\
   G_M-I
   \end{pmatrix}.

Its null space is the set of vectors unchanged by every operation. Let this
set be ``V_fixed``. The eigenvalue-one space of ``P`` is exactly the same
space. First, if ``\mathbf{x} \in V_{\mathrm{fixed}}``, each term in the
average is ``\mathbf{x}``, and therefore

.. math::

   P\mathbf{x}
   = \frac{1}{M}\sum_{g=1}^{M}G_g\mathbf{x}
   = \frac{1}{M}\sum_{g=1}^{M}\mathbf{x}
   = \mathbf{x}.

Conversely, suppose

.. math::

   P\mathbf{x} = \mathbf{x}

for some vector ``\mathbf{x}``. For any ``G_k``, the identity ``G_kP=P``
above gives

.. math::

   G_k\mathbf{x}
   = G_kP\mathbf{x}
   = P\mathbf{x}
   = \mathbf{x}.

Thus ``\mathbf{x}`` is unchanged by every symmetry and belongs to
``V_fixed``. In other words,

.. math::

   \ker(P-I) = \ker A = V_{\mathrm{fixed}}.

The direct method finds this space, usually through the null space of ``A``
(or the zero-eigenvalue space of ``A^T A``). Averaging is useful because it
also supplies the operator that symmetrizes an arbitrary vector:
``P\mathbf{x}`` is already in ``V_fixed``. For exact operations, its
idempotence also gives a particularly simple spectrum: eigenvalue ``1`` on
the allowed space and ``0`` on the rest. In contrast, finding the null space
through ``A^T A`` squares its numerical conditioning.

How a tensor becomes a vector
-----------------------------

A tensor has several indices. For example, a matrix has two indices:

.. math::

   A_{ij}.

The code temporarily puts all entries into one long vector:

.. math::

   \mathbf{x} =
   (A_{11}, A_{12}, A_{13}, A_{21}, \ldots)^T.

This is called *flattening*. It does not change the data; it only gives the
matrix a form on which ordinary matrix multiplication can be used.

For every symmetry operation, the code then builds one matrix ``G`` that
describes what happens to this flattened vector. Depending on the tensor, ``G``
can contain:

* rotations of Cartesian components;
* permutations of atom indices;
* both at the same time.

The implementation is split into two steps:

* :meth:`fd2bec.atomic.AtomicStructure.get_symmetry_operations` obtains the
  geometric rotations and translations;
* :meth:`fd2bec.atomic.AtomicStructure.get_tensor_symmetry_operations` turns
  them into matrices acting on the chosen tensor.

The latter returns matrices with the conceptual action

.. math::

   \mathbf{x}' = G\mathbf{x} + \mathbf{t}.

For an ordinary tensor, ``t`` is zero. For positions and other affine tensors,
it may be nonzero.

Affine quantities and the extra ``1``
------------------------------------------

Some quantities transform by a rotation *and* a shift:

.. math::

   \mathbf{x}' = G\mathbf{x} + \mathbf{t}.

This is not ordinary matrix multiplication yet. The code makes it linear by
adding a constant last component:

.. math::

   \widetilde{\mathbf{x}} =
   \begin{pmatrix}\mathbf{x}\\1\end{pmatrix},
   \qquad
   \widetilde{G} =
   \begin{pmatrix}G & \mathbf{t}\\0 & 1\end{pmatrix}.

Then

.. math::

   \widetilde{\mathbf{x}}'
   = \widetilde{G}\widetilde{\mathbf{x}}.

The extra ``1`` is not a physical tensor component. It is only a bookkeeping
device that lets the same matrix calculation handle translations.

In the source, this conversion is performed by
:func:`fd2bec.mathematics.affine2homogeneous`.

The code uses this construction for tensors with an affine axis. In
``get_symmetry_modes``, the extra row is removed again from the returned
``component_modes``.

What ``get_symmetry_modes`` returns
-----------------------------------

After constructing ``P``, the method solves an eigenvalue problem:

.. math::

   P\mathbf{v} = \lambda\mathbf{v}.

For an exact projection, the only eigenvalues are:

* ``1`` for allowed directions;
* ``0`` for forbidden directions.

The method keeps the eigenvectors with eigenvalue close to ``1``. Put these
vectors into the columns of the internal ``mode_basis``:

.. math::

   B = (\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_K).

Then every symmetry-allowed tensor can be written as

.. math::

   \mathbf{x}_{\mathrm{allowed}} = B\mathbf{c}.

The number ``K`` is the number of independent symmetry-allowed modes.

If a tensor value is supplied, the method finds its coefficients by solving

.. math::

   B\mathbf{c} \approx \mathbf{x}

with a least-squares solve. Thus the return values are:

``projection``
   The ``P`` matrix that projects arbitrary tensor components onto the
   symmetry-allowed subspace.

``mode_coefficients``
   The coefficients of the supplied tensor in that basis.

``component_modes``
   The same modes as rows, without the affine bookkeeping row. This is a
   convenient real-space view of the allowed modes.

For ``Position``, the affine coordinates themselves are a particular
symmetry-preserving reference configuration, not linear modes. Therefore,
``get_symmetry_modes(position)`` returns the normalized modes of
``Displacement`` about that reference. Use :meth:`fd2bec.atomic.AtomicStructure.symmetrize`
to obtain the symmetry-preserving position values themselves.

The method uses an eigenvalue tolerance ``atol`` because floating-point
arithmetic produces values such as ``0.999999999999`` instead of exactly
``1``.

The complete implementation path
---------------------------------

The calculation in the source follows this order:

1. :meth:`fd2bec.atomic.AtomicStructure.get_symmetry_operations` obtains the
   structure-preserving moves.
2. :meth:`fd2bec.atomic.AtomicStructure.get_tensor_symmetry_operations` adds
   tensor rotations, atom permutations, and affine shifts.
3. :meth:`fd2bec.atomic.AtomicStructure.get_symmetry_projection`
   averages the resulting matrices to form ``P``.
4. :meth:`fd2bec.atomic.AtomicStructure.get_symmetry_modes` finds the
   eigenvectors with eigenvalue ``1`` and solves for ``mode_coefficients``.

In short:

.. math::

   \boxed{
   \text{geometry} \;\longrightarrow\; G_g
   \;\longrightarrow\; P=\operatorname{average}(G_g)
   \;\longrightarrow\; B
   }

The result is a smaller coordinate system in which every coordinate already
respects the symmetry of the structure.

Where this is used
------------------

The same idea is used by several parts of the code:

* :func:`fd2bec.cli.displacements.generate_displacements.symmetry_inequivalent_displacements`
  uses the allowed modes to avoid redundant displacement calculations;
* :func:`fd2bec.piezoelectric.proper_piezoelectric_symmetry_basis` uses the
  modes as the allowed proper-piezoelectric tensor basis;
* :func:`fd2bec.system.LinearSystem` uses the symmetry-mode basis to reduce a Born
  effective-charge problem.

Useful tests are:

* ``tests/test_symmetrizer.py``;
* ``tests/test_symmetrizer_rotated.py``;
* ``tests/test_tensor_symmetry_operations.py``;
* ``tests/test_displacements.py``.
