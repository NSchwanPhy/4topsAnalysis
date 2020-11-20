# -*- encoding: utf-8 -*-
#
# Copyright (C) 2020 Ian Brock <Ian.Brock@cern.ch>
#
# This file is part of PyFeynHand.
#
# PyFeynHand is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# PyFeynHand is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyFeynHand; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA

"""Python wrapper for TikZ-FeynHand LaTeX package

PyFeynhandis a Python package helping in the creation of Feynman graphs
using the LaTeX package TikZ-FeynHand.
"""

__version__ = "0.1"

# Classes to make a FeynHand Feynamn graph
from .FeynHandTeX import FeynHandTeX
from .FeynHandVertex import Vertex
from .FeynHandPropagator import Propagator
from .FeynHandColor import colors

__all__ = ['FeynHandTeX', 'Vertex', 'Propagator', 'colors']
