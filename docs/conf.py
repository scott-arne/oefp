# OEFP Documentation Configuration
# Sphinx configuration for unified C++ and Python documentation.

from __future__ import annotations

import os
import shutil

# -- Path Setup ---------------------------------------------------------------

docs_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(docs_dir)

# -- Check for Doxygen --------------------------------------------------------

DOXYGEN_AVAILABLE = shutil.which("doxygen") is not None

if not DOXYGEN_AVAILABLE:
    print("WARNING: Doxygen not found. C++ API documentation will be skipped.")
    print("         Install doxygen to enable C++ documentation: brew install doxygen")

# -- Project Information ------------------------------------------------------

project = "OEFP"
copyright = "2026, OEFP Contributors"
author = "OEFP Contributors"
release = "0.2.7"
version = "0.2"

# -- General Configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

if DOXYGEN_AVAILABLE:
    extensions.insert(0, "breathe")
    extensions.insert(1, "exhale")

templates_path = ["_templates"]
exclude_patterns = ["_build", "_doxygen", "Thumbs.db", ".DS_Store"]

# -- Breathe Configuration (Doxygen -> Sphinx) --------------------------------

if DOXYGEN_AVAILABLE:
    breathe_projects = {"oefp": "_doxygen/xml"}
    breathe_default_project = "oefp"
    breathe_default_members = ("members", "undoc-members")

    exhale_args = {
        "containmentFolder": "./cpp-api",
        "rootFileName": "library_root.rst",
        "rootFileTitle": "C++ API Reference",
        "doxygenStripFromPath": "../include",
        "createTreeView": True,
        "exhaleExecutesDoxygen": True,
        "exhaleDoxygenStdin": """INPUT = ../include/oefp
RECURSIVE = YES
EXTRACT_ALL = YES
GENERATE_XML = YES
XML_OUTPUT = xml
GENERATE_HTML = NO
GENERATE_LATEX = NO
QUIET = YES
JAVADOC_AUTOBRIEF = YES
QT_AUTOBRIEF = YES""",
    }

# -- Napoleon Configuration ---------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc Configuration ----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Intersphinx Configuration ------------------------------------------------

intersphinx_mapping = {}
if os.environ.get("OEFP_DOCS_ONLINE") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
    }
intersphinx_timeout = 10

# -- HTML Output Configuration ------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
