#!/bin/sh
# Build the arXiv upload, build/arxiv.tar.gz, and prove it compiles on its own:
#   docker compose exec paper ./package_arxiv.sh
# - one flattened main.tex (latexpand), comments removed, bibliography inlined from the generated .bbl
#   (arXiv's own BibTeX run is not something to rely on);
# - only the figures the paper actually includes;
# - a clean-directory test build with pdflatex alone, as arXiv compiles it.
set -e
cd "$(dirname "$0")"
if grep -rn '\\todo{' main.tex abstract.tex sections/; then
	echo "Refusing to package: TODO markers remain (listed above)."
	exit 1
fi
./build.sh
rm -rf build/arxiv build/arxiv_test build/arxiv.tar.gz
mkdir -p build/arxiv
latexpand --empty-comments --expand-bbl build/main.bbl main.tex > build/arxiv/main.tex
for figure in $(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' build/arxiv/main.tex | sed 's/.*{\(.*\)}/\1/'); do
	mkdir -p "build/arxiv/$(dirname "$figure")"
	cp "$figure" "build/arxiv/$figure"
done
cp -r build/arxiv build/arxiv_test
(cd build/arxiv_test && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
grep -q -i "undefined" build/arxiv_test/main.log && { echo "Undefined references in the arXiv build:"; grep "undefined" build/arxiv_test/main.log; exit 1; }
tar -czf build/arxiv.tar.gz -C build/arxiv .
echo "OK: build/arxiv.tar.gz ($(du -h build/arxiv.tar.gz | cut -f1)), test build build/arxiv_test/main.pdf"
