#!/bin/sh
# Build the TMLR submission and what OpenReview takes with it, on the host (it needs git; LaTeX runs in the paper
# container):
#   app/latex_tmlr/package_tmlr.sh
# - build/submission.pdf: the anonymous paper (tmlr.sty without the accepted or preprint option);
# - build/supplement.zip: the code and the raw run outputs (without their PNG plots), from the committed tree, so
#   reviewers can regenerate every number (python src/tools/paper_results.py --paper latex_tmlr).
# Refuses while \todo{}s remain, while the supplement's paths have uncommitted changes, when the build has undefined
# references, citations or results, when anything identifies the authors, or when the supplement exceeds 100 MB.
set -e
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)
# Names, emails, affiliation, the public repository and the arXiv version: none may reach the submission.
IDENTIFYING='Vicco|Zarate|blasvicco|mvicco|Litoral|CONICET|Ciencias H|2611\.21598'
SUPPLEMENT_PATHS="README.md STANDARDS.md docker-compose.yml hub/python app/src app/tests app/requirements.txt app/pytest.ini app/.pylintrc app/cloud_manifest.txt app/outputs"
MAX_SUPPLEMENT_BYTES=100000000

refuse() {
	echo "Refusing to package: $1"
	exit 1
}

if grep -rn '\\todo{' main.tex abstract.tex sections/ deanonymized/; then
	refuse "TODO markers remain (listed above)."
fi
# shellcheck disable=SC2086
if [ -n "$(git -C "$ROOT" status --porcelain -- $SUPPLEMENT_PATHS)" ]; then
	git -C "$ROOT" status --short -- $SUPPLEMENT_PATHS | head -20
	refuse "the supplement is built from the committed tree, and these paths have uncommitted changes (listed above)."
fi

(cd "$ROOT" && docker compose exec -T paper latex_tmlr/build.sh >/dev/null)
if grep -E 'undefined|Undefined result' build/main.log; then
	refuse "the build has undefined references, citations or results (listed above)."
fi

# Every word of the PDF comes from these sources (deanonymized/ is read only by the accepted and preprint versions);
# the PDF's own bytes are checked for metadata, and its text when pdftotext is installed.
if grep -rniE "$IDENTIFYING" main.tex abstract.tex references.bib sections/ generated/; then
	refuse "the paper's sources identify the authors (listed above)."
fi
if LC_ALL=C grep -aqiE "$IDENTIFYING" build/main.pdf; then
	refuse "the PDF's bytes identify the authors."
fi
if command -v pdftotext >/dev/null && pdftotext build/main.pdf - | grep -iE "$IDENTIFYING"; then
	refuse "the PDF's text identifies the authors (listed above)."
fi

rm -f build/supplement.zip
# shellcheck disable=SC2086
git -C "$ROOT" archive --format=zip -o "$PWD/build/supplement.zip" HEAD -- $SUPPLEMENT_PATHS ':(exclude)app/outputs/**/*.png'
if unzip -Z1 build/supplement.zip | grep -iE "$IDENTIFYING" || unzip -p build/supplement.zip | LC_ALL=C grep -aqiE "$IDENTIFYING"; then
	zipgrep -iE "$IDENTIFYING" build/supplement.zip | head -20 || true
	refuse "the supplement identifies the authors (listed above)."
fi
SUPPLEMENT_BYTES=$(wc -c <build/supplement.zip | tr -d ' ')
if [ "$SUPPLEMENT_BYTES" -gt "$MAX_SUPPLEMENT_BYTES" ]; then
	refuse "the supplement is $SUPPLEMENT_BYTES bytes; TMLR takes up to 100 MB."
fi

cp build/main.pdf build/submission.pdf
echo "OK: build/submission.pdf ($(du -h build/submission.pdf | cut -f1)), build/supplement.zip ($(du -h build/supplement.zip | cut -f1)), from commit $(git -C "$ROOT" rev-parse --short HEAD)"
