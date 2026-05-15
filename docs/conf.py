# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Allow autodoc to find the package source.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "zarr-vectors-tools"
copyright = (
    "2024, BRIDGE Neuroscience. Aligned to the Zarr Vectors specification by "
    "Forrest Collman, Allen Institute for Brain Sciences."
)
author = "BRIDGE Neuroscience"
release = "0.2.0"

# -- General configuration ----------------------------------------------------
extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    # Markdown support
    "myst_parser",
    # API diagrams
    "sphinx.ext.graphviz",
    # Copy button on code blocks
    "sphinx_copybutton",
]

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",      # ::: directive syntax
    "deflist",          # definition lists
    "fieldlist",        # field lists
    "tasklist",         # - [ ] checkboxes
    "attrs_inline",     # inline attribute syntax
]
myst_heading_anchors = 3

# Napoleon (Google / NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# autosectionlabel — prefix with document name to avoid collisions
autosectionlabel_prefix_document = True

# intersphinx — link to upstream docs
intersphinx_mapping = {
    "python":       ("https://docs.python.org/3", None),
    "numpy":        ("https://numpy.org/doc/stable", None),
    "zarr":         ("https://zarr.readthedocs.io/en/stable", None),
    "zarr_vectors": ("https://zarr-vectors.readthedocs.io/en/latest", None),
}

# Source suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "zarr-vectors-tools"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary":    "#e0195c",
        "color-brand-content":    "#e0195c",
        "font-stack":             "'DM Sans', sans-serif",
        "font-stack--monospace":  "'JetBrains Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary":    "#ff72c0",
        "color-brand-content":    "#ff72c0",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/Andrew-Keenlyside/zarr-vectors-tools/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Show "Edit on GitHub" links
html_context = {
    "github_user":    "Andrew-Keenlyside",
    "github_repo":    "zarr-vectors-tools",
    "github_version": "main",
    "doc_path":       "docs",
}

# -- copybutton ---------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
