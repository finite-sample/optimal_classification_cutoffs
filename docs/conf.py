# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Load project information from pyproject.toml (single source of truth)
import pathlib

try:
    # Method 1: Try reading from installed package metadata
    from importlib.metadata import metadata

    pkg_meta = metadata("optimal-classification-cutoffs")
    project = pkg_meta["Name"]
    release = pkg_meta["Version"]
    # Extract author from Author-email field since new pyproject.toml format
    author_email = pkg_meta.get("Author-email", "")
    if author_email and "<" in author_email:
        author = author_email.split("<")[0].strip()
    else:
        author = pkg_meta.get("Author") or "Gaurav Sood"
except Exception:
    # Method 2: Fallback - read directly from pyproject.toml for development
    import tomllib

    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    project_info = pyproject_data["project"]
    project = project_info["name"]
    release = project_info["version"]
    author = project_info["authors"][0]["name"]

copyright = f"2024, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",  # Re-enabled for notebook execution and embedding
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = 'auto'  # Execute notebooks during build
nbsphinx_allow_errors = False  # Fail build if notebooks have errors
nbsphinx_kernel_name = 'python3'
nbsphinx_timeout = 300  # 5 minute timeout for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Custom CSS for notebooks
nbsphinx_custom_formats = {
    '.ipynb': ['nbsphinx', 'Jupyter Notebook'],
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/finite-sample/optimal-classification-cutoffs/",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1e40af",
        "color-admonition-background": "#f8fafc",
        "color-sidebar-background": "#f1f5f9",
        "color-sidebar-background-border": "#e2e8f0",
        "font-stack": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
        "color-admonition-background": "#1e293b",
        "color-sidebar-background": "#0f172a",
        "color-sidebar-background-border": "#334155",
    },
}

# Custom title and logo
html_title = "Optimal Classification Cutoffs"
html_short_title = "Optimal Cutoffs"

# -- Extension configuration -------------------------------------------------

# Napoleon settings - using NumPy style only (codebase standard)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Set up autodoc
autodoc_mock_imports = []
