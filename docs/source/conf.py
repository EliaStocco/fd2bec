from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "fd2bec documentation"
copyright = "fd2bec contributors"
author = "fd2bec contributors"

# Generated maps use these paired roles so identifiers remain exact while
# gaining natural line-breaking in both PDF and HTML output.
rst_prolog = r"""
.. role:: raw-latex(raw)
   :format: latex

.. role:: raw-html(raw)
   :format: html
"""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
]

templates_path = []
exclude_patterns = []

html_theme = "alabaster"

# Keep source documents consecutive in the PDF. Treating each RST file as a
# chapter would force a page break even when the preceding page is mostly empty.
latex_toplevel_sectioning = "section"

# Follow the restrained technical-manual style used by the FHI-aims manual:
# A4/10 pt, Latin Modern sans serif, simple headings, blue links, and compact
# running headers. Keep the setup self-contained so the existing
# Sphinx build remains reproducible.
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "extraclassoptions": "oneside,openany,titlepage,final",
    "geometry": r"\usepackage{geometry}",
    "fontpkg": r"\usepackage{lmodern}",
    "fncychap": "",
    "printindex": "",
    "preamble": r"""
\usepackage[small,sf]{caption}
\usepackage{booktabs}
\usepackage{bm}
\usepackage{enumitem}
\usepackage{placeins}

\definecolor{fdlinkblue}{rgb}{0,0,1}
\definecolor{fdheadingblue}{RGB}{31,78,107}
\PassOptionsToPackage{
  colorlinks=true,
  linkcolor=fdlinkblue,
  menucolor=fdlinkblue,
  citecolor=fdlinkblue,
  urlcolor=fdlinkblue,
  bookmarksnumbered=true,
  hyperindex=true
}{hyperref}

\AtBeginDocument{
  \hypersetup{
    pdfauthor={fd2bec contributors},
    pdftitle={fd2bec mathematical documentation},
    pdfsubject={Mathematics, package structure, tests, and dependencies}
  }
  \renewcommand{\familydefault}{\sfdefault}
  \sffamily
  \setlength{\parindent}{0pt}
  \setlength{\parskip}{0.65ex}
  \setlength{\headheight}{13pt}
  \setlength{\abovedisplayskip}{5pt plus 1pt minus 1pt}
  \setlength{\belowdisplayskip}{5pt plus 1pt minus 1pt}
  \setlength{\abovedisplayshortskip}{3pt plus 1pt}
  \setlength{\belowdisplayshortskip}{3pt plus 1pt}
  \renewcommand{\arraystretch}{0.95}
  \setlist{itemsep=0.15ex,topsep=0.45ex,parsep=0pt,partopsep=0pt}
  \newgeometry{left=1.8cm,right=1.8cm,top=1.7cm,bottom=1.8cm,headsep=0.45cm}

  \titleformat{\section}
    {\normalfont\sffamily\Large\bfseries\color{fdheadingblue}}
    {\thesection}{0.65em}{}
  \titlespacing*{\section}{0pt}{1.7ex plus 0.4ex minus 0.2ex}{0.6ex}
  \titleformat{\subsection}
    {\normalfont\sffamily\large\bfseries\color{fdheadingblue}}
    {\thesubsection}{0.6em}{}
  \titlespacing*{\subsection}{0pt}{1.2ex plus 0.3ex minus 0.2ex}{0.4ex}

  \fancypagestyle{normal}{
    \fancyhf{}
    \fancyhead[L]{\sffamily\slshape\nouppercase{\rightmark}}
    \fancyhead[R]{\sffamily\thepage}
  }
  \fancypagestyle{plain}{
    \fancyhf{}
    \fancyhead[R]{\sffamily\thepage}
    \renewcommand{\headrulewidth}{0pt}
  }
}
""",
    "maketitle": r"""
\hypersetup{pageanchor=false}
\begin{titlepage}
\sffamily
\thispagestyle{empty}
\vspace*{-2cm}

\begin{center}
\rule{\textwidth}{1pt}

\vspace{0.45cm}
{\Huge\bfseries Finite Differences to Born Effective Charges:\\[0.35cm] fd2bec\par}
\vspace{0.45cm}

\rule{\textwidth}{1pt}

\vspace{2.8cm}
{\LARGE Technical documentation\par}
\vspace{0.8cm}
{\Large Mathematics, package structure, tests, and dependencies\par}

\vfill
\rule{\textwidth}{1pt}
\vspace{0.45cm}

{\large fd2bec contributors\\
Manual compiled on \today\par}
\end{center}
\end{titlepage}
\hypersetup{pageanchor=true}
\setlength{\headheight}{13pt}
""",
}

latex_documents = [
    (
        "index",
        "fd2bec_math.tex",
        "fd2bec documentation",
        "fd2bec contributors",
        "howto",
    )
]

graphviz_output_format = "png"
