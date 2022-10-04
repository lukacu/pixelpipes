# -*- coding: utf-8 -*-
#
# PixelPipes documentation build configuration file

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

from pixelpipes import __version__

extensions = ['sphinx.ext.autodoc', 'link_roles', 'nodedoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

from recommonmark.parser import CommonMarkParser

source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = ['.rst', '.md']
source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'PixelPipes'
copyright = u'2022, Luka Cehovin Zajc'
author = u'Luka Cehovin Zajc'

version = __version__

# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/*setup*']

pygments_style = 'sphinx'

todo_include_todos = False

html_theme = 'sphinx_rtd_theme'

html_static_path = [] # html_static_path = ['_static']

htmlhelp_basename = 'pixelpipesdocs'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
     # The paper size ('letterpaper' or 'a4paper').
     #
     # 'papersize': 'letterpaper',

     # The font size ('10pt', '11pt' or '12pt').
     #
     # 'pointsize': '10pt',

     # Additional stuff for the LaTeX preamble.
     #
     # 'preamble': '',

     # Latex figure (float) alignment
     #
     # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'PixelPipes.tex', u'PixelPipes Documentation',
     u'Luka Cehovin Zajc', 'manual'),
]

man_pages = [
    (master_doc, 'pixelpipes', u'PixelPipes Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'PixelPipes', u'PixelPipes Documentation',
     author, 'PixelPipes', 'One line description of project.',
     'Miscellaneous'),
]

def setup(app):
    app.connect('autodoc-skip-member', skip_nodes)


def skip_nodes(app, what, name, obj, skip, options):
    from inspect import isclass
    from pixelpipes.graph import Node
    if what != "class":
        return None
    if not isclass(obj):
        return None
    return issubclass(obj, Node)
