def colors(BW):
    """Define standard colors"""

    if BW:
        oextra = '-BW'
        iCOLOR = oCOLOR = pCOLOR = vCOLOR = 'Black'
        dCOLOR = 'Black'
    else:
        oextra = ''
        iCOLOR = 'Blue'
        oCOLOR = 'IndianRed'
        pCOLOR = 'ForestGreen'
        vCOLOR = 'Crimson'
        dCOLOR = 'DarkCyan'

    return (iCOLOR, oCOLOR, pCOLOR, vCOLOR, dCOLOR)