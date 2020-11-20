#! /usr/bin/env python3

# from PyFeynHand import *
from PyFeynHand import FeynHandTeX
from PyFeynHand import Vertex
from PyFeynHand import Propagator
from PyFeynHand import colors
import argparse

# B&W or colour?
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--BW', dest='BW', action='store_true', help='B&W output')
parser.add_argument('-c', '--color', dest='BW', action='store_false', help='Colour output')
parser.add_argument('--colour', dest='BW', action='store_false', help='Colour output')
parser.add_argument('--pdf', action='store_true', help='Run pdflatex')
parser.set_defaults(BW=False)
args = parser.parse_args()

iCOLOR, oCOLOR, pCOLOR, vCOLOR, dCOLOR = colors(args.BW)
if args.BW:
    oextra = '-BW'
else:
    oextra = ''

#------------------------------------------------------------------------------
# s-channel gg->tt
fd1 = FeynHandTeX(vcolor=vCOLOR)
#l1 = Label(r'$\Pg{}\Pg \to \Pqt\Paqt', x=0, y=2)

pntIn = dict()
pntOut = dict()
pntIn['g1'] = Vertex(-3, +2,label=r'\Pg')
pntIn['g2'] = Vertex(-3, -2, label=r'\Pg')
pntOut['t1'] = Vertex(+3, +2, label=r'\Pqt')
pntOut['t2'] = Vertex(+3, -2, label=r'\Paqt')

vtx = dict()
vtx['gg'] = Vertex(-1, 0, vtype='dot')
vtx['gt'] = Vertex(+1, 0, vtype='dot')
# Make a dictionary with all vertices - must have unique keys
fd1.vertices = {**pntIn, **pntOut, **vtx}

fd1.propagators.append(Propagator('gluon', 'g1', 'gg', iCOLOR))
fd1.propagators.append(Propagator('gluon', 'g2', 'gg', iCOLOR))
fd1.propagators.append(Propagator('gluon', 'gg', 'gt', pCOLOR, r'\Pg'))
fd1.propagators.append(Propagator('fermion', 'gt', 't1', oCOLOR))
fd1.propagators.append(Propagator('fermion', 'gt', 't2', oCOLOR, reverse=True))

fd1.write('gg_ttbar_1' + oextra, compile=args.pdf)

#------------------------------------------------------------------------------
# t-channel gg->tt
fd2 = FeynHandTeX(vcolor=vCOLOR)
#l2 = Label(r'$\Pg{}\Pg \to \Pqt\Paqt', x=0, y=2)

pntIn = dict()
pntOut = dict()
pntIn['g1'] = Vertex(-2, +1.5, label=r'\Pg')
pntIn['g2'] = Vertex(-2, -1.5, label=r'\Pg')
pntOut['t1'] = Vertex(+2, +1.5, label=r'\Pqt')
pntOut['t2'] = Vertex(+2, -1.5, label=r'\Paqt')

vtx = dict()
vtx['gt1'] = Vertex(0, +1.5, vtype='dot')
vtx['gt2'] = Vertex(0, -1.5, vtype='dot')
# Make a dictionary with all vertices - must have unique keys
fd2.vertices = {**pntIn, **pntOut, **vtx}

fd2.propagators.append(Propagator('gluon', 'g1', 'gt1', iCOLOR))
fd2.propagators.append(Propagator('gluon', 'g2', 'gt2', iCOLOR))
fd2.propagators.append(Propagator('fermion', 'gt1', 't1', oCOLOR))
fd2.propagators.append(Propagator('fermion', 'gt2', 'gt1', pCOLOR, r'\Pqt'))
fd2.propagators.append(Propagator('fermion', 'gt2', 't2', oCOLOR, reverse=True))

fd2.write('gg_ttbar_2' + oextra, compile=args.pdf)

#------------------------------------------------------------------------------
# u-channel gg->tt
fd3 = FeynHandTeX(vcolor=vCOLOR)
#l3 = Label(r'$\Pg{}\Pg \to \Pqt\Paqt', x=0, y=2)

pntIn = dict()
pntOut = dict()
pntIn['g1'] = Vertex(-2, +1.5, label=r'\Pg')
pntIn['g2'] = Vertex(-2, -1.5, label=r'\Pg')
pntOut['t1'] = Vertex(+2, +1.5, label=r'\Pqt')
pntOut['t2'] = Vertex(+2, -1.5, label=r'\Paqt')

vtx = dict()
vtx['gt1'] = Vertex(0, +1.5, vtype='dot')
vtx['gt2'] = Vertex(0, -1.5, vtype='dot')
# Make a dictionary with all vertices - must have unique keys
fd3.vertices = {**pntIn, **pntOut, **vtx}

fd3.propagators.append(Propagator('gluon', 'g1', 'gt1', iCOLOR))
fd3.propagators.append(Propagator('gluon', 'g2', 'gt2', iCOLOR))
fd3.propagators.append(Propagator('fermion', 'gt2', 't1', oCOLOR))
fd3.propagators.append(Propagator('fermion', 'gt1', 'gt2', pCOLOR, r'\Pqt'))
fd3.propagators.append(Propagator('fermion', 'gt1', 't2', oCOLOR, reverse=True))

fd3.write('gg_ttbar_3' + oextra, compile=args.pdf)

#------------------------------------------------------------------------------
# qq->tt
fd4 = FeynHandTeX(vcolor=vCOLOR)
#l4 = Label(r'\Pep{}\Pem scattering', x=0, y=2)

pntIn = dict()
pntOut = dict()
pntIn['q1'] = Vertex(-3, +2, label=r'\Pq')
pntIn['q2'] = Vertex(-3, -2, label=r'\Paq')
pntOut['t1'] = Vertex(+3, +2, label=r'\Pqt')
pntOut['t2'] = Vertex(+3, -2, label=r'\Paqt')

vtx = dict()
vtx['qg'] = Vertex(-1, 0, vtype='dot')
vtx['gt'] = Vertex(+1, 0, vtype='dot')
# Make a dictionary with all vertices - must have unique keys
fd4.vertices = {**pntIn, **pntOut, **vtx}

fd4.propagators.append(Propagator('fermion', 'q1', 'qg', iCOLOR))
fd4.propagators.append(Propagator('fermion', 'q2', 'qg', iCOLOR, reverse=True))
fd4.propagators.append(Propagator('gluon', 'qg', 'gt', pCOLOR, r'\Pg'))
fd4.propagators.append(Propagator('fermion', 'gt', 't1', oCOLOR))
fd4.propagators.append(Propagator('fermion', 'gt', 't2', oCOLOR, reverse=True))

fd4.write('qq_ttbar' + oextra, compile=args.pdf)
