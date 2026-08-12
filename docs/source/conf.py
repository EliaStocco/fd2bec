from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "fd2bec mathematical documentation"
copyright = "fd2bec contributors"
author = "fd2bec contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = []
exclude_patterns = []

html_theme = "alabaster"

# Use LaTeX's standard fonts so the PDF can be built with a small TeX install.
latex_elements = {
    "fontpkg": "",
}

latex_documents = [
    (
        "index",
        "fd2bec_math.tex",
        "fd2bec mathematical documentation",
        "fd2bec contributors",
        "manual",
    )
]
