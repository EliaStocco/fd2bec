import os

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from fd2bec import SYMPREC

API_KEY = os.getenv("MP_API_KEY")

output_dir = "spacegroup_structures"
os.makedirs(output_dir, exist_ok=True)

best_structures = {}

# Tolerances
SYMPREC_LOOSE = SYMPREC   # recover MP symmetry
SYMPREC_TIGHT = 1e-1 * SYMPREC_LOOSE   # enforce strict symmetry

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        is_stable=True,
        is_metal=False,
        fields=["material_id", "structure", "symmetry"],
        chunk_size=16,
    )

    for doc in docs:
        target_sg = doc.symmetry.number
        structure = doc.structure
        material_id = doc.material_id
        n_sites = len(structure)

        # --- Step 1: check if we can recover the nominal SG (loose tolerance)
        sga_loose = SpacegroupAnalyzer(structure, symprec=SYMPREC_LOOSE)
        detected_loose = sga_loose.get_space_group_number()

        if detected_loose != target_sg:
            continue  # discard: not even approximately correct

        # --- Step 2: refine structure (snap to symmetry)
        refined = sga_loose.get_refined_structure()

        # --- Step 3: verify at tight tolerance
        sga_tight = SpacegroupAnalyzer(refined, symprec=SYMPREC_TIGHT)
        detected_tight = sga_tight.get_space_group_number()

        if detected_tight != target_sg:
            # symmetry not robust → discard
            continue

        # --- Step 4: keep best (smallest)
        if (
            target_sg not in best_structures
            or n_sites < best_structures[target_sg][0]
        ):
            best_structures[target_sg] = (n_sites, material_id, refined)
            print(f"Accepted SG {target_sg}: {material_id} ({n_sites} atoms)")

# --- Write results
for sg_number, (n_sites, material_id, structure) in best_structures.items():
    filename = f"SG_{sg_number}_{material_id}.cif"
    filepath = os.path.join(output_dir, filename)

    writer = CifWriter(structure)
    writer.write_file(filepath)

    print(f"Saved SG {sg_number} -> {filename}")

print(f"\nCollected {len(best_structures)} space groups.")