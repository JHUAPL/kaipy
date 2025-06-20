# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../kaipy'))
sys.path.insert(1, os.path.abspath('../../kaipy/scripts/quicklook'))
sys.path.insert(2, os.path.abspath('../../kaipy/scripts/preproc'))
sys.path.insert(3, os.path.abspath('../../kaipy/scripts/postproc'))
sys.path.insert(4, os.path.abspath('../../kaipy/scripts/OHelio'))
sys.path.insert(5, os.path.abspath('../../kaipy/scripts/datamodel'))
os.environ["KAIPYHOME"] = os.path.abspath('../../kaipy')

# -- Project information -----------------------------------------------------

project = 'Kaipy'
copyright = '2023 - JHU/APL, NSF NCAR, and Rice University'
author = 'Kaiju Team'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.autoprogram',
    'sphinx_rtd_theme'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = '_static/kaipy-logo.png'

html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'collapse_navigation': False,
    'navigation_depth': 4,
}

html_css_files = [
    'css/sidebar_theme.css',
]

# Mock Imports
autodoc_mock_imports = ['cartopy']