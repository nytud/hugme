# -- Project information -----------------------------------------------------

project = 'Hungarian Generative Model Evaluation benchmark'
copyright = '2025, HUN-REN Hungarian Research Centre for Linguistics'
author = 'Ligeti-Nagy Noémi'
# release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = ["myst_parser", "sphinx_rtd_theme"]

templates_path = ['_templates']
exclude_patterns = []

language = 'hu'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo' # sphinx_rtd_theme
html_static_path = ['_static']

html_logo = "logo.png"
html_title = "HuGME"
