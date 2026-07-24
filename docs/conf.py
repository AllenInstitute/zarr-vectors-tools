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
# Package version. Independent of the on-disk FORMAT version this package
# targets, which is ZVF 0.9.0 (the merged links/<delta>/<offsets>/ layout).
release = "0.2.0"
version = release

# On-disk format targeted by this release. Single source of truth for the
# format version quoted throughout the prose.
#
# It has to be wired up twice, because the two parsers do not share a
# substitution mechanism: rst_prolog covers the .rst pages (|zvf_version|),
# myst_substitutions covers the .md pages ({{ zvf_version }}). rst_prolog
# alone leaves the literal "|zvf_version|" in the rendered Markdown.
zvf_format_version = "0.9.0"

rst_prolog = f"""
.. |zvf_version| replace:: {zvf_format_version}
"""

myst_substitutions = {
    "zvf_version": zvf_format_version,
}

# -- General configuration ----------------------------------------------------
extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
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
    "substitution",     # {{ zvf_version }} - see myst_substitutions above
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

# zarr-vectors isn't on PyPI; mock it so autodoc can import this package
# without needing the runtime dependency installed.
autodoc_mock_imports = ["zarr_vectors"]

# autosectionlabel — prefix with document name to avoid collisions
autosectionlabel_prefix_document = True

# intersphinx — link to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "zarr":   ("https://zarr.readthedocs.io/en/stable", None),
    # zarr-vectors does not yet publish an objects.inv on Read the Docs.
    # Add a "zarr_vectors" entry here once that site exists.
}

# extlinks - canonical outbound targets, so the three sibling sites are
# spelled once here rather than pasted into forty pages.
#
#   :zvpy:`getting_started/concepts.html`  -> main library docs
#   :zvspec:`05-zarr-store-structure.html` -> the specification site
#   :zvrepo:`blob/main/schema/README.md`   -> the library source repo
extlinks = {
    "zvpy":   ("https://zarr-vectors-py.readthedocs.io/en/latest/%s", "%s"),
    "zvspec": ("https://alleninstitute.github.io/zarr_vectors/%s", "%s"),
    "zvrepo": ("https://github.com/BRIDGE-Neuroscience/zarr-vectors-py/%s", "%s"),
}
extlinks_detect_hardcoded_links = False

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
    "source_repository": "https://github.com/AllenInstitute/zarr-vectors-tools/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Show "Edit on GitHub" links
html_context = {
    "github_user":    "AllenInstitute",
    "github_repo":    "zarr-vectors-tools",
    "github_version": "main",
    "doc_path":       "docs",
}

# -- copybutton ---------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
