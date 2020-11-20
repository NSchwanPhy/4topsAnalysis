import pyx

# Get my extra functions
from .draw_stuff import draw_fermion, draw_boson, draw_gluon, draw_higgs

__all__ = ['draw_fermion', 'draw_boson', 'draw_gluon', 'draw_higgs']

# Define font and colours for LaTeX
# defaulttexrunner has to be replaced by defaulttextengine with PyX 0.15
# Similar fixes are also necessary in the PyFeyn package
# pyx.text.defaulttexrunner.preamble(r"\usepackage[varg]{txfonts}")
pyx.text.defaulttexrunner.preamble(r"\usepackage{newtxtext}")
pyx.text.defaulttexrunner.preamble(r"\usepackage{newtxmath}")
# pyx.text.defaulttexrunner.preamble(r"\usepackage{mathpazo}")
# Sans serif fonts
# pyx.text.defaulttexrunner.preamble(r"\usepackage{sansmathfonts}")
# pyx.text.defaulttexrunner.preamble(r"\usepackage[scaled=0.95]{helvet}")
# pyx.text.defaulttexrunner.preamble(r"\renewcommand{\rmdefault}{\sfdefault}")
pyx.text.defaulttexrunner.preamble(r"\usepackage[T1]{fontenc}")
pyx.text.defaulttexrunner.preamble(r"\usepackage[dvipsnames]{xcolor}")
# Use this line if hepnicenames removed from pyfeyn/__init__.py
# pyx.text.defaulttexrunner.preamble(r"\usepackage[italic]{heppennames}")
# Adjust the kerning for bar over particles with narrow glyphs (from Andy Buckley)
# This is probably only needed with the italic option
pyx.text.defaulttexrunner.preamble(r"\makeatletter\def\@shiftlen@anti@gen@bar{0mu}\makeatother")
# Increase default text size
pyx.text.defaulttexrunner.preamble(r"\usepackage[fontsize=12pt]{scrextend}")
