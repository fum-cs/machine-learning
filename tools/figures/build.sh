#!/usr/bin/env bash
# Compile the TikZ network figures and write the PNGs used by the notebooks.
# Requires: pdflatex + pdftoppm (TeX Live).  Usage: bash tools/figures/build.sh
set -euo pipefail
cd "$(dirname "$0")"

out="../../notebooks/img"
for f in nn-5 nn-6 nn-6b; do
  pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" > "$f.build.log"
  pdftoppm -png -r 220 -singlefile "$f.pdf" "$out/$f"
  echo "wrote $out/$f.png"
done
rm -f ./*.aux ./*.log ./*.build.log ./*.pdf
