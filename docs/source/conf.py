import os
import sys
from datetime import datetime
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python_utils")))

project   = "LagRBF"
author    = "Anthony Jourdon"
copyright = f"{datetime.now().year}, {author}"
release   = "1.0.0"

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.napoleon",
  "sphinx.ext.mathjax",
  "sphinx.ext.intersphinx",
  "sphinx.ext.viewcode",
  #"breathe",
  #"myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
  'prev_next_buttons_location': 'both',
  #'style_external_links': True,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

#breathe_projects = {
#  "lagrbf_c": os.path.abspath(
#    os.path.join(os.path.dirname(__file__), "..", "build", "doxygen", "xml")
#  )
#}
#breathe_default_project = "lagrbf_c"
primary_domain = "c"
highlight_language = "c"

breathe_domain_by_extension = {
  "h": "c",
  "c": "c",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "pyvista": ("https://docs.pyvista.org/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None)
}

rst_prolog = """
.. _PETSc: https://petsc.org/release/
.. _DM: https://petsc.org/release/manualpages/DM/DM/
.. _PetscErrorCode: https://petsc.org/release/manualpages/Sys/PetscErrorCode/
.. _PetscInt: https://petsc.org/release/manualpages/Sys/PetscInt/
.. _PetscReal: https://petsc.org/release/manualpages/Sys/PetscReal/
"""

def _run_cdocs_parser(app):
  docs_source = Path(__file__).resolve().parent
  repo_root = docs_source.parent.parent
  parser = docs_source / "cdocs_parser.py"
  out_dir = docs_source / "api" / "c" / "generated"
  index_file = out_dir / "index.rst"

  cmd = [
      sys.executable,
      str(parser),
      "--src",
      str(repo_root / "src"),
      str(repo_root / "include"),
      "--out",
      str(out_dir),
      "--index",
      str(index_file),
  ]

  subprocess.run(cmd, check=True)

def setup(app):
  app.connect("builder-inited", _run_cdocs_parser)