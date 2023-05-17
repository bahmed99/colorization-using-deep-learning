#!/bin/bash

pdflatex cahier-des-charges.tex
bibtex cahier-des-charges
pdflatex cahier-des-charges.tex
pdflatex cahier-des-charges.tex