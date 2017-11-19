all: generate open


open:
	open report.pdf
	open -a iTerm.app

generate:
	pdflatex report.tex
	bibtex report.aux
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f *.aux *.pdf *.log *.bbl *.blg
