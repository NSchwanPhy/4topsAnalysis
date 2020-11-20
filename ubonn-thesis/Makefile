# Makefile to create a new thesis skeleton
# Can also run feynmf/feynmp/tikz on files in a directory
#
# Note that "make update" can overwrite this file

THESIS  = mythesis
TEXLIVE = 2016
# TEXOLD = 2009
# EXTRACMD = --shell-escape
ifndef FEYNDIR
FEYNDIR = feynmf
endif
ifndef FEYNFILES
FEYNFILES = $(wildcard $(FEYNDIR)/*.tex)
endif
ifndef TIKZDIR
TIKZDIR = tikz
endif
ifndef PYFEYNDIR
PYFEYNDIR = pyfeyn
endif
ifndef PYFEYNHANDDIR
PYFEYNHANDDIR = pyfeynhand
endif
ifndef TIKZFILES
TIKZFILES = $(wildcard $(TIKZDIR)/*.tex)
endif
ifndef PYFEYNFILES
PYFEYNFILES = $(wildcard $(PYFEYNDIR)/*.py)
endif
ifndef PYFEYNHANDFILES
PYFEYNHANDFILES = $(wildcard $(PYFEYNHANDDIR)/*.py)
endif
ifdef file
FEYNFILES = $(FEYNDIR)/$(file).tex
TIKZFILES = $(TIKZDIR)/$(file).tex
PYFEYNFILES = $(PYFEYNDIR)/$(file).py
PYFEYNHANDFILES = $(PYFEYNHANDDIR)/$(file).py
endif
ifndef AWKDIR
AWKDIR = .
endif
ifndef MPOSTDIR
MPOSTDIR = mpost_tmp
endif
LATEXMK = latexmk -pdf
# LATEXMK = latexmk -e '$$pdflatex=q/pdflatex %O -shell-escape %S/' -pdf $(GUIDE)

.PHONY: skelcopy \
	feynmf feynmp tikz pyfeyn pyfeynhand \
	cleanfeynmf cleanfeynmp cleantikz cleanpyfeyn cleanpictpdf \
	help test

# New thesis
new: skelcopy
	#cp thesis_skel/thesis_skel.tex        $(THESIS)/$(THESIS).tex
	sed 's/texlive=2016/texlive=$(TEXLIVE)/' thesis_skel/thesis_skel.tex > $(THESIS)/$(THESIS).tex

# New thesis with astrophysics style of references
astro: skelcopy
	sed 's/texlive=2016/texlive=$(TEXLIVE)/' thesis_skel/thesis_astro_skel.tex > $(THESIS)/$(THESIS).tex
	cp thesis_skel/thesis_astro_intro.tex  $(THESIS)/thesis_intro.tex

# New thesis with old version of TeX Live
# new09: skelcopy
# 	#cp thesis_skel/thesis_2009_skel.tex  $(THESIS)/$(THESIS).tex
# 	sed 's/texlive=2009/texlive=$(TEXOLD)/' thesis_skel/thesis_2009_skel.tex > $(THESIS)/$(THESIS).tex

skelcopy:
	mkdir $(THESIS)
	mkdir $(THESIS)/bib
	mkdir $(THESIS)/figs
	cp thesis_skel/thesis_defs.sty        $(THESIS)/
	cp thesis_skel/thesis_intro.tex       $(THESIS)/
	cp thesis_skel/thesis_appendix.tex    $(THESIS)/
	cp thesis_skel/thesis_acknowledge.tex $(THESIS)/
	cp thesis_skel/thesis_cv.tex          $(THESIS)/
	cp thesis_skel/cover_only.tex         $(THESIS)/
	cp ubonn-thesis.sty                   $(THESIS)/
	cp ubonn-biblatex.sty                 $(THESIS)/
	cp thesis_skel/Makefile               $(THESIS)/
	cp thesis_skel/thesis_refs.bib        $(THESIS)/bib/
	cp refs/standard_refs-biber.bib       $(THESIS)/bib/
	cp -R figs/cover                      $(THESIS)/figs/

update:
	-cp -i ubonn-thesis.sty                $(THESIS)/
	-cp -i ubonn-biblatex.sty              $(THESIS)/
	-cp -i thesis_skel/Makefile            $(THESIS)/
	-cp -i refs/standard_refs-biber.bib    $(THESIS)/bib/
	-cp -i -R figs/cover                   $(THESIS)/

feynmf: $(FEYNFILES)
	@echo "Feymf Feynman graph files: $^"
	-rm feynmf_files.inp
	touch feynmf_files.inp
	$(foreach feynfile, $^, echo $(feynfile) >>feynmf_files.inp;)
	cat feynmf_files.inp
	cat feynmf_files.inp | awk -f $(AWKDIR)/awk_feynmf >feynmf_all.tex
	-pdflatex feynmf_all
	$(foreach feynfile, $^, $(AWKDIR)/run_mf $(feynfile);)
	pdflatex feynmf_all; pdflatex feynmf_all; pdflatex feynmf_all

feynmp: $(FEYNFILES)
	@echo "Feynmp Feynman graph files: $^"
	$(foreach feynfile, $^, $(AWKDIR)/run_mp $(feynfile);)

tikz: $(TIKZFILES)
	@echo "TikZ Feynman graph files: $^"
	(cd $(TIKZDIR); $(foreach tikzfile, $^, pdflatex $(notdir $(tikzfile));))

pyfeyn: $(PYFEYNFILES)
	@echo "PyFeyn Feynman graph files: $^"
	(cd $(PYFEYNDIR); $(foreach pyfeynfile, $^, ./$(notdir $(pyfeynfile));))

# Make color and B&W PDF versions of TikZ-FeynHand graphs
pyfeynhand: $(PYFEYNHANDFILES)
	@echo "PyFeynHand Feynman graph files: $^"
	(cd $(PYFEYNHANDDIR); $(foreach pyfeynhandfile, $^, ./$(notdir $(pyfeynhandfile)) --pdf;))
	(cd $(PYFEYNHANDDIR); $(foreach pyfeynhandfile, $^, ./$(notdir $(pyfeynhandfile)) --pdf --BW;))

cleanall: clean cleanpictpdf

clean: cleanfeynmf cleanfeynmp cleantikz cleanpyfeynhand

cleanfeynmf:
	-rm *.mf *.tfm *.t1 *.600gf *.600pk *.log
	-rm feynmf_all.* feynmf_files.inp

cleanfeynmp:
	-rm $(MPOSTDIR)/*.1   $(MPOSTDIR)/*.log
	-rm $(MPOSTDIR)/*.mp  $(MPOSTDIR)/*.t1
	-rm $(MPOSTDIR)/*.aux
	-rm $(MPOSTDIR)/*_mp.tex $(MPOSTDIR)/*_mp.pdf
	-rmdir $(MPOSTDIR)

cleantikz:
	-rm $(TIKZDIR)/*.log $(TIKZDIR)/*.aux

cleanpyfeynhand:
	-rm $(PYFEYNHANDDIR)/*.aux $(PYFEYNHANDDIR)/*.log

cleanpictpdf:
	-rm $(FEYNDIR)/*.pdf
	-rm $(TIKZDIR)/*.pdf
	-rm $(PYFEYNDIR)/*.pdf
	-rm $(PYFEYNHANDDIR)/*.pdf
	-rm $(PYFEYNHANDDIR)/*.tex

help:
	@echo "Possible commands:"
	@echo "new   [THESIS=dirname] [TEXLIVE=YYYY]: create a new thesis skeleton"
	@echo "astro [THESIS=dirname] [TEXLIVE=YYYY]: create a new astrophysics thesis skeleton"
	@echo "  Default TeX Live version is 2016"
	# @echo "new09 [THESIS=dirname] [TEXOLD=YYYY]: create a new thesis skeleton"
	# @echo "  Default old TeX Live version is 2009"
	@echo "update [THESIS=dirname]: update to a newer version of ubonn-thesis"
	@echo "feynmf: run feynmf for all .tex files in $(FEYNDIR)"
	@echo "feynmp: run feynmp for all .tex files in $(FEYNDIR)"
	@echo "tikz:   run tikz for all .tex files in $(TIKZDIR)"
	@echo "pyfeyn: run Python for all .py files in $(PYFEYNDIR)"
	@echo "pyfeynhand: run Python for all .py files in $(PYFEYNHANDDIR)"
	@echo "cleanfeynmf: clean up feynmf output files"
	@echo "cleanfeynmp: clean up feynmp output files"
	@echo "cleantikz:   clean up tikz auxiliary output files"
	@echo "cleanpyfeynhand: clean up pyfeynhand auxiliary output files"

test:
	@echo "Feynmf Feynman graphs dir: $(FEYNDIR)"
	@echo "Feynmf Feynman graphs files: $(FEYNFILES)"
	@echo "TikZ   Feynman graphs dir: $(TIKZDIR)"
	@echo "TikZ   Feynman graphs files: $(TIKZFILES)"
	@echo "PyFeyn Feynman graphs dir: $(PYFEYNDIR)"
	@echo "PyFeyn Feynman graphs files: $(PYFEYNFILES)"
	@echo "PyFeynHand Feynman graphs dir: $(PYFEYNHANDDIR)"
	@echo "PyFeynHand Feynman graphs files: $(PYFEYNHANDFILES)"
