import os
import numpy as np

from fd2bec.io import read, write

INPUT_DIR = "sg_prototypes"
OUTPUT_DIR = "sg_prototypes_rotated"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def random_rotation_matrix():
    """Uniform random 3D rotation matrix."""
    rand = np.random.normal(size=(3, 3))
    q, _ = np.linalg.qr(rand)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_shift():
    """Small fractional shift."""
    return np.random.uniform(-0.2, 0.2, size=3)


success = 0

for sg in range(1, 231):

    infile = os.path.join(INPUT_DIR, f"SG_{sg}.cif")

    try:
        atoms = read(infile)

        # -------------------------
        # 1. random rotation
        # -------------------------
        alpha, beta, gamma =  np.random.normal(size=(3,))

        positions = atoms.get_positions()
        center = positions.mean(axis=0)

        # atoms.positions = (positions - center) @ R.T + center
        atoms.rotate(alpha,'x',center=center,rotate_cell=True)
        atoms.rotate(beta,'y',center=center,rotate_cell=True)
        atoms.rotate(gamma,'z',center=center,rotate_cell=True)

        # -------------------------
        # 2. random shift
        # -------------------------
        shift = random_shift()
        atoms.translate(shift)

        # -------------------------
        # 3. write output
        # -------------------------
        outfile = os.path.join(OUTPUT_DIR, f"SG_{sg}_aug.extxyz")
        write(outfile, atoms)

        print(f"✅ SG {sg} rotated + shifted")
        success += 1

    except Exception as e:
        print(f"❌ SG {sg} failed: {e}")

print(f"\nDone: {success}/230")