import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "protein-design-tools"
copyright = "2024, Andrew Schaub"
author = "Andrew Schaub"
# Single source of truth for the version is pyproject.toml.
with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")) as _f:
    release = re.search(r'^version = "([^"]+)"', _f.read(), re.MULTILINE).group(1)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Handles automatic documentation
    "sphinx.ext.napoleon",  # Supports Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.autosummary",  # Generates summary tables for modules/classes
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
