import os
import numpy as np
from ase import Atoms
from ase.io import write

os.makedirs("point_group_dataset", exist_ok=True)

def save(name, atoms):
    write(f"point_group_dataset/{name}.xyz", atoms)

# ----------------------------
# Helper functions
# ----------------------------
def circle_points(n, r=1.0, z=0.0):
    pts = []
    for i in range(n):
        theta = 2*np.pi*i/n
        pts.append((r*np.cos(theta), r*np.sin(theta), z))
    return pts

# ----------------------------
# LOW SYMMETRY
# ----------------------------

# C1
save("C1_CHBrClF",
     Atoms(["C","H","Br","Cl","F"],
           positions=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(0.3,0.4,0.5)]))

# Cs (one mirror plane: xy-plane)
save("Cs_CH2ClBr",
     Atoms(["C","H","H","Cl","Br"],
           positions=[(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]))

# Ci (inversion)
coords = [(0.7,0.2,0.1), (-0.7,-0.2,-0.1)]
save("Ci_dummy",
     Atoms(["He","He"], positions=coords))

# ----------------------------
# CYCLIC GROUPS
# ----------------------------

# C2 (twisted H2O2-like)
save("C2_H2O2_like",
     Atoms(["O","O","H","H"],
           positions=[(-0.7,0,0),(0.7,0,0),(0,0.8,0.6),(0,-0.8,-0.6)]))

# C2v (water)
save("C2v_H2O",
     Atoms(["O","H","H"],
           positions=[(0,0,0),(0.958,0,0),(-0.239,0.927,0)]))

# C3v (ammonia)
pts = [(0,0,0)] + circle_points(3, r=1.0, z=0.3)
save("C3v_NH3", Atoms(["N","H","H","H"], positions=pts))

# C4v (square pyramid)
pts = [(0,0,0)] + circle_points(4, r=1.2, z=0) + [(0,0,1.2)]
save("C4v_square_pyramidal",
     Atoms(["X","F","F","F","F","F"], positions=pts))

# C3h (planar trigonal + mirror)
pts = [(0,0,0)] + circle_points(3, r=1.2, z=0)
save("C3h_BF3_like", Atoms(["B","F","F","F"], positions=pts))

# ----------------------------
# DIHEDRAL GROUPS
# ----------------------------

# D2h (ethylene)
save("D2h_C2H4",
     Atoms(["C","C","H","H","H","H"],
           positions=[
               (-0.67,0,0),(0.67,0,0),
               (-1.2,0.9,0),(-1.2,-0.9,0),
               (1.2,0.9,0),(1.2,-0.9,0)
           ]))

# D3h (BF3)
pts = [(0,0,0)] + circle_points(3, r=1.3)
save("D3h_BF3", Atoms(["B","F","F","F"], positions=pts))

# D4h (square planar XeF4)
pts = [(0,0,0)] + circle_points(4, r=1.5)
save("D4h_XeF4", Atoms(["Xe","F","F","F","F"], positions=pts))

# D3d (ethane staggered)
top = circle_points(3, r=1.0, z=0.7)
bottom = circle_points(3, r=1.0, z=-0.7)
save("D3d_ethane",
     Atoms(["C","C"] + ["H"]*6,
           positions=[(0,0,0.7),(0,0,-0.7)] + top + bottom))

# D2d (allene-like)
save("D2d_allene",
     Atoms(["C","C","C","H","H","H","H"],
           positions=[
               (0,0,0),(0,0,1.3),(0,0,-1.3),
               (1,0,1.8),(-1,0,1.8),
               (0,1,-1.8),(0,-1,-1.8)
           ]))

# ----------------------------
# IMPROPER
# ----------------------------

# S4 (twisted square)
square = circle_points(4, r=1.2)
twisted = [(x, y, 0.3*(-1)**i) for i,(x,y,_) in enumerate(square)]
save("S4_distorted_square", Atoms(["X"]*4, positions=twisted))

# ----------------------------
# LINEAR
# ----------------------------

save("Cinfv_CO",
     Atoms("CO", positions=[(0,0,0),(0,0,1.13)]))

save("Dinfh_N2",
     Atoms("N2", positions=[(0,0,-0.55),(0,0,0.55)]))

# ----------------------------
# CUBIC GROUPS
# ----------------------------

# Td (CH4)
a = 1.1
save("Td_CH4",
     Atoms(["C","H","H","H","H"],
           positions=[
               (0,0,0),
               ( a, a, a),
               (-a,-a, a),
               (-a, a,-a),
               ( a,-a,-a)
           ]))

# Oh (SF6)
a = 1.6
save("Oh_SF6",
     Atoms(["S"] + ["F"]*6,
           positions=[
               (0,0,0),
               ( a,0,0),(-a,0,0),
               (0, a,0),(0,-a,0),
               (0,0, a),(0,0,-a)
           ]))

print("Dataset generated!")
