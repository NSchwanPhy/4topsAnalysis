import shlex
import subprocess
import sys

class FeynHandTeX:
    """Base class for creating FeynHand LaTeX file"""

    def __init__(self, font='newtx', linewidth='1.0pt', lcolor=None, pcolor=None, vcolor=None):
        """Initialize FeynHand TeX object"""

        self.font = font
        self.linewidth = linewidth
        self.lcolor = lcolor
        self.pcolor = pcolor
        self.vcolor = vcolor
        self.vertices = dict()
        self.propagators = list()

    def __str__(self):
        s = []
        s.append('FeynHandTeX object')
        s.append('Label color: {}'.format(str(self.lcolor)))
        s.append('Propagator color: {}'.format(str(self.pcolor)))
        s.append('Vertex color: {}'.format(str(self.vcolor)))
        for key, vtx in self.vertices.items():
            s.append(r'Vertex {} {} {} {} {} {} {}'.format(
                key, vtx.getType(), vtx.getX(), vtx.getY(),
                vtx.getLabel(), vtx.getVertexColor(), vtx.getLabelColor()))
        for prop in self.propagators:
            s.append(r'Propagator {} From {} to {} {} {} {}'.format(
                prop.getType(), prop.getVertex1(), prop.getVertex2(),
                prop.getLabel(), prop.getPropColor(), prop.getLabelColor()))

        return '\n'.join(s)

    def open(self, file):
        "Open output TeX file"
        fout = open(file + '.tex', 'w')
        return fout

    def preamble(self):
        "Return a list of string with the preamble"
        self.preamble = list()
        self.preamble.append(r'\documentclass{standalone}')
        self.preamble.append(r'\usepackage[utf8]{inputenc}')
        if self.font == 'newtx':
            self.preamble.append(r'\usepackage{newtxtext}')
            self.preamble.append(r'\usepackage{newtxmath}')
        else:
            self.preamble.append(r'\usepackage{' + self.font + r'}')
        self.preamble.append(r'\usepackage[italic]{hepnicenames}')
        # Tweak to get decent overline width for slim letters
        self.preamble.append(r'\makeatletter\def\@shiftlen@anti@gen@bar{0mu}\makeatother')
        self.preamble.append(r'\usepackage[svgnames]{xcolor}')
        self.preamble.append(r'\usepackage{tikz-feynhand}')
        self.preamble.append('\n')
        self.preamble.append(r'\begin{document}')
        self.preamble.append(r'\begin{tikzpicture}')
        self.preamble.append(r'  \setlength{{\feynhandlinesize}}{{{}}}'.format(self.linewidth))
        if self.pcolor:
            self.preamble.append(r'  \tikzfeynhandset{every fermion={/tikz/color=' + self.pcolor + r'},}')
            self.preamble.append(r'  \tikzfeynhandset{every boson={/tikz/color=' + self.pcolor + r'},}')
        if self.vcolor:
            self.preamble.append(r'  \tikzfeynhandset{every dot={/tikz/color=' + self.vcolor + r'},}')
        self.preamble.append(r'  \begin{feynhand}')
        
        return self.preamble

    # def write(self, vertices, propagators, file):
    def write(self, file, compile=False):
        "Write Feynman graph to file"
        # print('Opening', file)
        fout = self.open(file)

        # Preamble
        for line in self.preamble():
            fout.write(line + '\n')

        # Feynman graph
        # Vertices
        # print('Vertices', vertices)
        for key, vtx in self.vertices.items():
            fout.write(r'    \vertex [{}'.format(vtx.getType()))
            _vcolor = None
            # Use label colour for particle vertices
            if vtx.getType() == 'particle':
                if vtx.getLabelColor():
                    _vcolor = vtx.getLabelColor()
                elif self.lcolor:
                    _vcolor = self.lcolor
                else:
                    _vcolor = self.getPropColor(key)
            # Use vertex colour for all other vertices
            else:
                if vtx.getVertexColor():
                    _vcolor = vtx.getVertexColor()
                elif self.vcolor:
                    _vcolor = self.vcolor
            if _vcolor:
                fout.write(r', {}'.format(_vcolor))
            fout.write(r'] ({}) at ({}, {}) {{'.format(
                key, vtx.getX(), vtx.getY()))
            if vtx.getLabelColor():
                fout.write(r'\color{{{}}}'.format(vtx.getLabelColor()))
            elif self.lcolor:
                fout.write(r'\color{{{}}}'.format(self.lcolor))
            fout.write(r'{}}}'.format(vtx.getLabel()))
            fout.write(';\n')
        # Propagators
        # print('Propagators', propagators)
        for prop in self.propagators:
            pline = r'    \propagator [{}'.format(prop.getType())
            if prop.getPropColor():
                pline += ', {}'.format(prop.getPropColor())
            elif self.pcolor:
                pline += ', {}'.format(self.pcolor)
            pline += r'] ({}) to'.format(prop.getVertex1())
            if prop.getLabel():
                if prop.getLabelColor():
                    pline += r' [edge label={}, color={}]'.format(
                        prop.getLabel(), prop.getLabelColor())
                elif self.lcolor:
                    pline += r' [edge label={}, color={}]'.format(
                        prop.getLabel(), self.lcolor)
                else:
                    pline += r' [edge label={}, color={}]'.format(
                        prop.getLabel(), prop.getPropColor())
            pline += r' ({})'.format(prop.getVertex2())
            fout.write(pline + ';\n')

        # Terminate Feynman graph
        fout.write(r'  \end{feynhand}' + '\n')
        fout.write(r'\end{tikzpicture}' + '\n')
        fout.write(r'\end{document}' + '\n')

        fout.close()

        # Run pdlalatex
        if compile:
            cmdstr = 'pdflatex ' + file
            cmd =  shlex.split(cmdstr)
            print('pdflatex command:', cmd)
            process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):
                sys.stdout.write(line.decode(sys.stdout.encoding, errors='ignore'))

        return

    def getPropColor(self, vkey):
        "Return the propagator color associated with vertex vkey"

        # See if there is a propagator with this vertex and its colour is set
        # If no color set return default colour
        for prop in self.propagators:
            if vkey == prop.getVertex1() or vkey == prop.getVertex2():
                if prop.getPropColor():
                    return prop.getPropColor()
                else:
                    return self.pcolor
        
        return None
        