# PyFeynHand

Author: Ian C. Brock (ian.brock@cern.ch)

Feynman diagrams using LaTeX TikZ-FeynHand package with a Python wrapper.

This package provides a wrapper around the LaTeX package TikZ-FeynHand for drawing Feynman graphs.
It creates a LaTeX file and optionally runs `pdflatex` on the file.

The advantages of `PyFeynHand` are:
* You can use Python to script Feynman graphs, while making direct use of LaTeX.
* You can use the same fonts as are used in ATLAS documents.
* You can use particle names defined by the `hepparticle`, `heppennames` and/or `hepnicenmaes` packages.
* You have the full power of `TikZ` behind you to do fancy things!
* You can either `\input` the LaTeX file or the resulting PDF files in your document.

The LaTeX file uses the `standalone` document class,
so can be `\input` directly into another LaTeX file.
You just have to include the `standalone` package in your document.
You can of course input the PDF file instead using `includegraphics`.

It would probably be a good idea to be able to add extra LaTeX code at various places in the LaTeX file,
but this has not yet been implemented.

## Colours

You can control the colours of both labels and lines.
By default the label colours will be the same as the external line or propagator colour.
Create the Feynman graph object `FeynHandTeX` with the option `lcolor='Black'` if you want black labels.

When you initialise the `FeynHandTeX` object you can specify:
* `font`: font package to use (default is newtx)
* `lcolor`: colour of vertex and propagator labels
* `pcolor`: colour of propagator line
* `vcolor`: colour of vertex dot

When you create a vertex or propagator, you can also specify the colours of these three elements.
These override the default colours.

In the scripts I define a series of colours that are used:
* `iCOLOR`: incoming lines
* `oCOLOR`: outgoing lines
* `pCOLOR`: propagators
* `vCOLOR`: vertices
* `dCOLOR`: decay products, e.g. lepton from Z->ll decay

The defaults are set in `PyFeynHand/FeynHandColor.py`.
You can adjust these colours if you want different ones and also use them for whatever purpose you like.

## Further information

If you want black and white versions of the graphs, pass the option `--BW` or `-b` to the script.
The filenames then get `BW` appended.
If you want the PDF file to be created. pass the option `--pdf` to the script.

If you want to use the package to make your own Feynman graphs, note that
the `PyFeynHand` directory has to be linked from the directory with the Feynman graphs.

The default setting uses the `newtx` fonts, which are the default fonts for ATLAS documents.
Other fonts can be used by passing the package name you want to use to the `FeynHandTeX` object,
e.g. `font='txfonts'`.

The default line thickness can be changed by passing `linewidth` to the `FeynHandTeX` object.
The default setting in PyFeynHand is `linewidth=1.0pt`.
The TikZ-FeynHand uses as default to `0.5pt`, which I find very thin.
It may be better to use an even larger value for slides.
However, changing the line width also changes the amplitude and frequency of photon and gluon oscillations.

