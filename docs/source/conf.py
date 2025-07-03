# Configuration file for the Sphinx documentation builder.

# -- Project information
import sys
import os


sys.path.insert(0, os.path.abspath('../..'))
# print current path
print(os.path.abspath('../..'))
project = 'STADiffuser'
copyright = '2025, Chihao Zhang'
author = 'Chihao Zhang'

release = '0.1'
version = '0.1.0'


# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinx_design",
    "nbsphinx",
    "sphinxarg.ext",
]

# autoapi_dirs = ["autoapi"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'furo'
# set hthe
html_logo = "_static/logo.png"
html_title = "STADiffuser documentation"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}
# -- Options for EPUB output
autodoc_mock_imports = ["scanpy", "torch", "torch_geometric", "scipy", "pandas"]
html_static_path = ['_static']
epub_show_urls = 'footnote'
autosummary_generate = True

