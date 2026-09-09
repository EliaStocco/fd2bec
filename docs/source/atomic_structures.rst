Atomic structures and atom matching
===================================

The :class:`fd2bec.atomic.AtomicStructure` class is the common structural
representation used by the symmetry and finite-difference machinery. It
normalizes ASE structures, keeps Cartesian and fractional coordinates
together, and defines when two structures contain the same atoms at the same
sites even when their atom ordering differs.

One representation for molecules and crystals
-----------------------------------------------

An atomic structure contains four pieces of physical information:

``symbols``
   The chemical symbol associated with each atom index.

``positions``
   An ``(N, 3)`` array of Cartesian positions.

``cell``
   Three lattice vectors, stored as an :class:`ase.cell.Cell` for a periodic
   structure.

``frac_pos``
   An ``(N, 3)`` array of fractional positions for a periodic structure.

The periodicity flag ``pbc`` decides how these fields are interpreted. Only
fully periodic and fully non-periodic ASE objects are accepted. Mixed
boundary conditions such as ``(True, True, False)`` are deliberately rejected
by :meth:`fd2bec.atomic.AtomicStructure.from_ase`.

For a periodic structure, fd2bec uses the ASE row-vector convention. If
``A`` is the ``(3, 3)`` cell matrix, fractional and Cartesian row vectors are
related by

.. math::

   \mathbf{r}_i = \mathbf{s}_i A,

where :math:`\mathbf{s}_i` is a fractional position and
:math:`\mathbf{r}_i` is its Cartesian position. The constructor accepts
either representation and calculates the other. If both are supplied, it
checks this relation within the global numerical tolerance ``ATOL``.

For a molecule, ``cell`` is represented internally by a matrix of ``NaN``
values and ``frac_pos`` is consequently not a meaningful coordinate system.
Molecular geometry is always interpreted through ``positions`` in Cartesian
coordinates.

Construction from ASE
---------------------

The usual entry point is

.. code-block:: python

   structure = AtomicStructure.from_ase(atoms)

The conversion performs the following normalization:

1. read the chemical symbols without changing their order;
2. copy the selected ASE position array (``positions`` by default);
3. require either three periodic directions or none;
4. retain the cell only for a periodic structure;
5. derive the missing Cartesian or fractional representation.

The optional ``keyword`` argument can select another ``atoms.arrays`` entry
as the coordinates. This is useful when a workflow stores a reference or
predicted geometry alongside the ordinary ASE positions.

``from_ase`` copies the ASE positions and cell. After construction, the
coordinate arrays and cell array are marked read-only. This prevents
accidental in-place modification of the geometry used by cached symmetry
results. ``AtomicStructure`` is not a frozen Python dataclass, however, so the
intended way to create a modified structure is the
:meth:`fd2bec.atomic.AtomicStructure.clone` method rather than mutating an
existing instance.

Cloning and coordinate consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``clone`` uses :func:`dataclasses.replace` and then runs the same normalization
again. Supplying new Cartesian positions discards the old fractional
positions and recomputes them; supplying new fractional positions does the
opposite. For example,

.. code-block:: python

   displaced = structure.clone(positions=new_positions)

produces a new structure with coordinates consistent with the original cell.
This pattern is used when testing symmetry operations and when constructing
finite displacements.

What structural equality means
------------------------------

``structure_a == structure_b`` delegates to
:meth:`fd2bec.atomic.AtomicStructure.is_equal_to`. Equality does not require
the atoms to have the same array order. Instead, fd2bec asks whether a
one-to-one, same-species atom assignment exists within a tolerance.

The comparison proceeds in stages:

1. both structures must have the same periodicity;
2. they must contain the same set and count of each chemical species;
3. their cell matrices must be numerically equal, with ``NaN`` cells treated
   as equal for two molecules;
4. atoms are matched independently inside each species;
5. every matched positional difference must be within ``atol``.

Consequently, atom order is irrelevant but the chosen periodic cell is not.
Two cells related by a lattice rotation, a different conventional-cell
choice, or a supercell transformation are not considered equal merely
because they describe the same infinite crystal. They must first be brought
to the same cell representation.

Species-resolved assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each species :math:`\alpha`, let :math:`a_i` be positions from ``self``
and :math:`b_j` positions from ``other``. The code builds a cost matrix

.. math::

   C^{(\alpha)}_{ij} = \left\lVert a_i-b_j\right\rVert

and uses :func:`scipy.optimize.linear_sum_assignment` to find the one-to-one
assignment with minimum total cost. Matching species separately prevents, for
example, an oxygen atom from being assigned to a nearby hydrogen atom.

For molecules, :math:`a_i` and :math:`b_j` are Cartesian positions and
``atol`` is therefore measured in Angstrom. The comparison is direct: fd2bec
does not automatically align or translate two molecules before matching
them.

For periodic structures, the cost is evaluated in fractional coordinates.
Each component of the difference is first wrapped into the nearest periodic
image:

.. math::

   \operatorname{wrap}(\Delta s)
   = (\Delta s + \tfrac{1}{2}) \bmod 1 - \tfrac{1}{2}.

The periodic cost is thus

.. math::

   C^{(\alpha)}_{ij}
   = \left\lVert
       \operatorname{wrap}(\mathbf{s}_i-\mathbf{s}'_j)
     \right\rVert.

Here ``atol`` is a fractional-coordinate tolerance. The current
implementation uses the Euclidean norm of the fractional components; it does
not multiply the difference by the cell metric when constructing the
assignment cost.

Mapping convention and reordering
---------------------------------

:meth:`fd2bec.atomic.AtomicStructure.get_atoms_mapping` exposes the mapping
used by equality. Its convention is

.. math::

   \mathtt{mapping}[i]
   = \text{index in ``self`` corresponding to atom }i\text{ in ``other``}.

The public method first gives clear errors for different atom counts,
periodicity, or chemical compositions. The internal mapping then verifies
that the result is a permutation of all atom indices.

:meth:`fd2bec.atomic.AtomicStructure.reordered_like` uses the inverse of this
mapping to return a new structure whose atom at index ``i`` corresponds to
atom ``i`` in a reference structure. The coordinates still come from the
structure being reordered. This distinction is important in finite-
difference datasets: reordering establishes correspondence without replacing
the calculated geometry.

Standardized periodic cells
---------------------------

Periodic structures also expose the representations used by ``spglib``:

.. math::

   (A, \{\mathbf{s}_i\}, \{Z_i\}),

namely the cell, fractional positions, and atomic numbers. From this tuple,
``spglib`` provides the cached symmetry dataset, the space-group number, and
a standardized cell. The :attr:`fd2bec.atomic.AtomicStructure.conventional`
property converts that standardized result back into an
``AtomicStructure``.

Standardization and equality answer different questions. Standardization
chooses a conventional periodic representation; equality checks whether two
already normalized structures match in their present representation.

Summary
-------

The structural layer can be summarized as

.. math::

   \boxed{
   \begin{gathered}
   \text{ASE structure}
   \longrightarrow
   \text{normalized coordinates}\\
   \longrightarrow
   \text{species-resolved assignment}
   \longrightarrow
   \text{atom correspondence}
   \end{gathered}
   }.

That correspondence is also what allows a symmetry operation to exchange
equivalent atoms. The next chapter builds the common symmetry formalism used
for both molecules and solids.
