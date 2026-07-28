This folder contains scripts useful to compute Born Effective Charges as derivatives of polarization or total dipole with respect to nuclear displacements.

Use `build_dataset4dPdR.py` for periodic structures and polarization data. Use
`build_dataset4dPdR_nonperiodic.py` for isolated structures (`pbc=False`) and
total-dipole data (for example `-f aims_dipole`). Both produce an extxyz file
that can be passed to `dPdR2bec.py`.
