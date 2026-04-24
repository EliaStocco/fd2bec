import os

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

API_KEY = os.getenv("MP_API_KEY")

# Output folder
output_dir = "spacegroup_structures"
os.makedirs(output_dir, exist_ok=True)

# Store best structure per space group
# sg_number -> (n_sites, material_id, structure)
best_structures = {}

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        is_stable=True,
        is_metal=False,
        # band_gap=(0.1, None),
        fields=["material_id", "structure", "symmetry"],
        chunk_size=16,
    )

    for doc in docs:
        sg_number = doc.symmetry.number
        structure = doc.structure
        n_sites = len(structure)
        material_id = doc.material_id

        # If new SG OR smaller structure found → replace
        if sg_number not in best_structures or n_sites < best_structures[sg_number][0]:
            best_structures[sg_number] = (n_sites, material_id, structure)
            # print(f"Updated SG {sg_number}: {material_id} ({n_sites} atoms)")

# Write final structures
for sg_number, (n_sites, material_id, structure) in best_structures.items():
    filename = f"SG_{sg_number}_{material_id}.cif"
    filepath = os.path.join(output_dir, filename)

    writer = CifWriter(structure)
    writer.write_file(filepath)

    print(f"Saved SG {sg_number} -> {filename}")

print(f"\nCollected {len(best_structures)} space groups.")
