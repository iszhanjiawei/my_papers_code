#!/bin/sh
set -eu
paper_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$paper_dir"
main_file=${1:-bare_jrnl_new_sample4.tex}
engine=${LATEX_ENGINE:-pdflatex}
case "$engine" in
  pdflatex) engine_flag=-pdf ;;
  xelatex) engine_flag=-xelatex ;;
  lualatex) engine_flag=-lualatex ;;
  *) echo "Unknown LATEX_ENGINE: $engine" >&2; exit 2 ;;
esac
exec /zjw524/bin/latexmk "$engine_flag" -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir=build "$main_file"
