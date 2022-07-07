%.pdf: %.md reference.bib
	pandoc $< -o $@ --pdf-engine=tectonic --bibliography=reference.bib --natbib
