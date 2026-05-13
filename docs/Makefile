# Minimal makefile for Sphinx documentation.

PYTHON       ?= python3
SPHINXOPTS   ?=
SPHINXBUILD  ?= $(PYTHON) -m sphinx
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help clean clean-generated html check

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" "_doxygen" "cpp-api"

clean-generated:
	rm -rf "_doxygen" "cpp-api"

html: clean-generated
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

check: clean-generated
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS)
