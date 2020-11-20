class Vertex:
    """Base class for all pointlike objects in Feynman diagrams."""

    def __init__(self, x, y, vtype='particle', label='', vcolor=None, lcolor=None):
        """Constructor."""
        self.setXY(x, y)
        # self.setBlob(blob)
        self.vtype = vtype
        self.label = label
        self.vcolor = vcolor
        self.lcolor = lcolor

    def __str__(self):
        s = []
        s.append(r'Vertex {} {} {} {} {}'.format(
            self.vtype, self.getX(), vtx.getY(), self.label,
            self.vcolor, self.lcolor))

        return '\n'.join(s)

    def setXY(self, xpos, ypos):
        "Set the x and y coordinates"
        self.setX(float(xpos))
        self.setY(float(ypos))
        return self

    def setX(self, x):
        "Set the x-coordinate"
        self.xpos = x
        return self

    def setType(self, vtype):
        "Set the type of vertex"
        self.vtype = vtype
        return self

    def setLabel(self, label):
        "Set the label"
        self.label = label
        return self

    def setVertexColor(self, vcolor):
        "Set the vertex color"
        self.vcolor = vcolor
        return self

    def setLabelColor(self, lcolor):
        "Set the label color"
        self.lcolor = lcolor
        return self

    def setY(self, y):
        "Set the y-coordinate"
        self.ypos = y
        return self

    def getX(self):
        "Return the x-coordinate"
        return self.xpos

    def getY(self):
        "Return the y-coordinate"
        return self.ypos

    def getXY(self):
        "Return the x and y coordinates as a 2-tuple."
        return self.getX(), self.getY()

    def getType(self):
        "Return the type of vertex"
        return self.vtype

    def getLabel(self):
        "Return the label"
        return self.label

    def getVertexColor(self):
        "Return the vertex color"
        return self.vcolor

    def getLabelColor(self):
        "Return the label color"
        return self.lcolor

