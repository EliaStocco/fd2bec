# Mathematical documentation

Part I explains how fd2bec represents and compares atomic structures, how it
uses one affine symmetry formalism for molecules and periodic solids, and the
mathematics behind `AtomicStructure.get_symmetrizer`. Part II contains
automatically generated package, CLI, pytest, and import-dependency maps.

Build the PDF with Sphinx and a LaTeX installation:

```bash
./tools/build_docs.sh
```

The PDF is written to `docs/_build/latex/`.
The script also creates the root-level symlink `fd2bec_math.pdf`.

`tools/build_docs.sh` runs `tools/generate_code_map.py` before Sphinx. Adding a
module below `fd2bec/`, a `[project.scripts]` entry, or a pytest reference is
therefore reflected in the next build. CLI modules without an explicit
README/RST reference are still listed and marked as undocumented.
