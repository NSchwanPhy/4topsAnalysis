from pyfeyn.lines import Fermion
from pyfeyn.lines import Photon
from pyfeyn.lines import Gluon
from pyfeyn.lines import Higgs
from pyfeyn.points import Point, Vertex
from pyfeyn.deco import Label
# Add other thicknesses if needed
from pyfeyn.paint import NORMAL, THICK3

nfermion = -1
nboson = -1
ngluon = -1
nhiggs = -1
fermions = []
bosons = []
gluons = []
higgs = []
# Default line thickness
myTHICK = THICK3
DEBUG = 0

def draw_fermion(pnt1, pnt2, COLOR, label, \
    displacement=-0.25, position=0.5, thickness=myTHICK, arrpos=0.53, bend=0, \
    reverse=False):
    global nfermion, fermions

    nfermion += 1
    if reverse:
        fermions.append(Fermion(pnt2, pnt1).addArrow(position=arrpos))
    else:
        fermions.append(Fermion(pnt1, pnt2).addArrow(position=arrpos))
    if label != '':
        if displacement == 0.0 and position == 0.0:
            if DEBUG >= 2: print('Not moving fermion label')
            fermions[nfermion].addLabel(label)
        else:
            if DEBUG >= 2: print('Moving fermion label by', displacement)
            if DEBUG >= 2: print('Position of fermion label is', position)
            fermions[nfermion].addLabel(label, displace=displacement, pos=position)
    fermions[nfermion].setStyles(COLOR)
    if thickness != NORMAL:
        if DEBUG >= 2: print('Changing fermion thickness to', thickness)
        fermions[nfermion].addStyle(thickness)
    if bend != 0:
        if DEBUG >= 2: print('Changing fermion bend to', bend)
        fermions[nfermion].bend(bend)


def draw_boson(pnt1, pnt2, COLOR, label,
    amplitude=0, frequency=0,
    displacement=-0.25, position=0.5, thickness=myTHICK, bend=0):
    global nboson, bosons

    nboson += 1
    bosons.append(Photon(pnt1, pnt2))
    if label != '':
        if displacement == 0.0 and position == 0.0:
            if DEBUG >= 2: print('Not moving boson label')
            bosons[nboson].addLabel(label)
        else:
            if DEBUG >= 2: print('Moving boson label by', displacement)
            if DEBUG >= 2: print('Position of boson label is', position)
            bosons[nboson].addLabel(label, displace=displacement, pos=position)
    bosons[nboson].setStyles(COLOR)
    if amplitude != 0:
        if DEBUG >= 2: print('Boson amplitude is', bosons[nboson].getAmplitude())
        if DEBUG >= 2: print('Changing boson amplitude to', amplitude)
        bosons[nboson].setAmplitude(amplitude)
    if frequency != 0:
        if DEBUG >= 2: print('Boson frequency is', bosons[nboson].getFrequency())
        if DEBUG >= 2: print('Changing boson frequency to', frequency)
        bosons[nboson].setFrequency(frequency)
    if thickness != NORMAL:
        if DEBUG >= 2: print('Changing boson thickness to', thickness)
        bosons[nboson].addStyle(thickness)
    if bend != 0:
        if DEBUG >= 2: print('Changing boson bend to', bend)
        bosons[nboson].bend(bend)


def draw_gluon(pnt1, pnt2, COLOR, label,
    amplitude=0, frequency=0,
    displacement=-0.25, position=0.5, thickness=myTHICK, bend=0):
    global ngluon, gluons

    ngluon += 1
    gluons.append(Gluon(pnt1, pnt2))
    if label != '':
        if displacement == 0.0 and position == 0.0:
            if DEBUG >= 2: print('Not moving gluon label')
            gluons[ngluon].addLabel(label)
        else:
            if DEBUG >= 2: print('Moving gluon label by', displacement)
            if DEBUG >= 2: print('Position of gluon label is', position)
            gluons[ngluon].addLabel(label, displace=displacement, pos=position)
    gluons[ngluon].setStyles(COLOR)
    if amplitude != 0:
        if DEBUG >= 2: print('Gluon amplitude is', bosons[nboson].getAmplitude())
        if DEBUG >= 2: print('Changing gluon amplitude to', amplitude)
        gluons[ngluon].setAmplitude(amplitude)
    if frequency != 0:
        if DEBUG >= 2: print('Gluon frequency is', bosons[nboson].getFrequency())
        if DEBUG >= 2: print('Changing gluon frequency to', frequency)
        gluons[ngluon].setFrequency(frequency)
    if thickness != NORMAL:
        if DEBUG >= 2: print('Changing gluon thickness to', thickness)
        gluons[ngluon].addStyle(thickness)
    if bend != 0:
        if DEBUG >= 2: print('Changing gluon bend to', bend)
        gluons[ngluon].bend(bend)


def draw_higgs(pnt1, pnt2, COLOR, label,
    displacement=-0.25, position=0.5, thickness=myTHICK, bend=0):
    global nhiggs, higgs

    nhiggs += 1
    higgs.append(Higgs(pnt1, pnt2))
    if label != '':
        if displacement == 0.0 and position == 0.0:
            if DEBUG >= 2: print('Not moving Higgs label')
            higgs[nhiggs].addLabel(label)
        else:
            if DEBUG >= 2: print('Moving Higgs label by', displacement)
            if DEBUG >= 2: print('Position of Higgs label is', position)
            higgs[nhiggs].addLabel(label, displace=displacement, pos=position)
    higgs[nhiggs].setStyles(COLOR)
    if thickness != NORMAL:
        if DEBUG >= 2: print('Changing Higgs thickness to', thickness)
        higgs[nhiggs].addStyle(thickness)
    if bend != 0:
        if DEBUG >= 2: print('Changing Higgs bend to', bend)
        higgs[nhiggs].bend(bend)
