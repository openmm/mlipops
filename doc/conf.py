# -*- coding: utf-8 -*-

import os
import sys
import importlib.metadata
import git


sys.path.append(os.path.abspath("../"))


# version specified in ../setup.py
version = importlib.metadata.version("mlipops")

repo = git.Repo(search_parent_directories=True)
short_sha = hash = repo.git.rev_parse(repo.head, short=True)

# get the the current tag if this commit has one
tag = next((tag for tag in repo.tags if tag.commit == repo.head.commit), None)

if tag is None:
    version_match = "dev"
    version = version_match
else:
    version_match = str(tag)
    version = version_match

print("version:", version)
print("git tag:", tag)
print("git sha:", short_sha)
print("version_match", version_match)


extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

myst_enable_extensions = ["dollarmath"]

source_suffix = ".rst"
master_doc = "index"

project = "MLIPOps"
copyright = "2026 Stanford University and the Authors"


exclude_patterns = ["_build", "_templates"]
# html_static_path = ["_static"]
templates_path = ["_templates"]

pygments_style = "sphinx"

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/openmm/mlipops",
}

html_sidebars = {
    "userguide": [],
    "**": ["sidebar-collapse", "sidebar-nav-bs"]
}


# settings for version switcher and warning
# html_theme_options["navbar_start"] = ["navbar-logo", "version-switcher"]
# html_theme_options["switcher"] = {
#     "json_url": "https://openmm.github.io/mlipops/dev/_static/versions.json",
#     "version_match": version_match,
# }

# https://github.com/pydata/pydata-sphinx-theme/issues/1552
html_theme_options["show_version_warning_banner"] = False
html_theme_options["check_switcher"] = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
