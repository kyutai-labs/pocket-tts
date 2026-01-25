import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "Pocket TTS"
copyright = "2025, Kyutai"
author = "Kyutai"
release = "1.0.2"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Source code linking
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Output directory
build_dir = Path(__file__).parent / "_build"
html_output_dir = build_dir / "html"
