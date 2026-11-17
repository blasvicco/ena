#!/bin/sh
# Build the paper into build/main.pdf, inside the TeX Live container:
#   docker compose exec paper ./build.sh
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
