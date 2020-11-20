#! /usr/bin/env python3

# Make Feynman graphs with PyFeyn and Python 3

from pyfeyn.diagrams import FeynDiagram
from pyfeyn.points import Point, Vertex, CIRCLE
from pyfeyn.deco import Label
from pyfeyn.paint import BLUE, FORESTGREEN, MAGENTA, RED, VIOLET
from myPyFeyn import draw_boson, draw_fermion, draw_gluon, draw_higgs

uCOLOR = [BLUE]
bCOLOR = [BLUE]
tCOLOR = [RED]
dCOLOR = [RED]
ZCOLOR = [RED]
pCOLOR = [FORESTGREEN]
eCOLOR = [BLUE]
gCOLOR = [VIOLET]

#-------------------------------------------------------------------------
# tZq production (5FS) - WWZ vertex
fd = FeynDiagram()
# l1 = Label('Lowest Order', x=0, y=2)

pntIn = []
pntOut = []
pntIn.append(Point(-3, +1.5))
pntIn.append(Point(-3, -1.5))
pntOut.append(Point(+3, +1.5))
pntOut.append(Point(+3, -1.5))

vtx = []
vtx.append(Vertex(0, +1.5, mark=CIRCLE, fill=[RED]))
vtx.append(Vertex(0, -1.5, mark=CIRCLE, fill=[RED]))

# Add Z boson on various lines
vtx.append(Vertex(0.5, 0.0, mark=CIRCLE, fill=[RED]))
pntOut.append(Point(+3, 0.0))

draw_fermion(pntIn[0], vtx[0], uCOLOR, r'\Pqu', displacement=+0.2)
draw_fermion(vtx[0], pntOut[0], dCOLOR, r'\Pqd', displacement=+0.2)
draw_boson(vtx[2], vtx[0], pCOLOR, r'\PW')
draw_boson(vtx[1], vtx[2], pCOLOR, r'\PW')
draw_fermion(pntIn[1], vtx[1], bCOLOR, r'\Pqb')
draw_fermion(vtx[1], pntOut[1], tCOLOR, r'\Paqt')
draw_boson(vtx[2], pntOut[2], ZCOLOR, r'\PZ')

fileout = 'tZq_5FS_feyn.pdf'
fd.draw(fileout)

#-------------------------------------------------------------------------
# tZq production (4FS) - radiation from one of the quarks
bCOLOR = [MAGENTA]
fd = FeynDiagram()
# l1 = Label('Lowest Order', x=0, y=2)

pntIn = []
pntOut = []
pntIn.append(Point(-4, +1.5))
pntIn.append(Point(-4, -1.5))
pntOut.append(Point(+3, +1.5))
pntOut.append(Point(+3, -1.0))

vtx = []
vtx.append(Vertex(-1.5, -1.5, mark=CIRCLE, fill=[RED]))
vtx.append(Vertex(0, +1.5, mark=CIRCLE, fill=[RED]))
vtx.append(Vertex(0, -1.0, mark=CIRCLE, fill=[RED]))

# Add Z boson on various lines
displacet = -0.25
vtx.append(Vertex(+0.75, -1.0, mark=CIRCLE, fill=[RED]))
pntOut.append(Point(3.0, 0.25))
displacet = 0.2
displaceZ = -0.2
pntOut.append(Point(+1.0, -2.5))

draw_fermion(pntIn[0], vtx[1], uCOLOR, r'\Pqu', displacement=-0.2)
draw_fermion(vtx[1], pntOut[0], dCOLOR, r'\Pqd', displacement=-0.2)
draw_boson(vtx[1], vtx[2], pCOLOR, r'\PW')
draw_gluon(pntIn[1], vtx[0], gCOLOR, '')
draw_fermion(vtx[0], vtx[2], bCOLOR, r'\Pqb')
draw_fermion(pntOut[3], vtx[0], bCOLOR, r'\Paqb')
draw_fermion(vtx[2], pntOut[1], tCOLOR, r'\Pqt', displacement=displacet)
draw_boson(vtx[3], pntOut[2], ZCOLOR, r'\PZ', displacement=displaceZ)

fileout = 'tZq_4FS_feyn.pdf'
fd.draw(fileout)
