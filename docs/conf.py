"""Sphinx configuration for dscompanion.

Doc build tooling only — never imported by dscompanion's runtime code. Build
locally with ``make html`` (see docs/Makefile).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import dscompanion

project = "dscompanion"
copyright = "2026, dscompanion contributors"
author = "dscompanion contributors"
release = dscompanion.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ── autodoc / autosummary ──────────────────────────────────────────────────
# Deliberately omits "members": True — autosummary's generated stub pages
# (autosummary_generate=True) already explicitly list every member via their
# own template (a "Methods" rubric + per-method automethod directives).
# Setting "members": True here as well documents __init__ and friends twice
# on the same page ("duplicate object description" warnings).
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True
autosummary_imported_members = False

# ── napoleon (Google-style docstrings — see ~/.claude/python_rules.md) ────
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

# ── HTML output ─────────────────────────────────────────────────────────────
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"dscompanion {release} — Internal Documentation"
