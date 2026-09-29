#!/usr/bin/env bash
# Stages docs/book/ (and only that) into a clean Jekyll source directory: the chapters, the figures, a generated
# _config.yml, the search index and the search UI include. It does not run Jekyll.
#
#   scripts/build_book_src.sh [DIR] [URL BASEURL]
#
# DIR defaults to _book_src. URL and BASEURL are written into _config.yml when given: the site workflow passes
# https://goinfer.dev and /book, so the theme's relative_url links (assets, the search index) resolve under /book/.
# Without them the config is the one GitHub Pages builds with (book-pages.yml), whose baseurl comes from the repo.
#
# The steps below were lifted verbatim from book-pages.yml so the two builds cannot disagree; their comments explain
# the traps (figures, relative links, the Liquid and comment guards, the non-vacuity check on the index).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-_book_src}"
BOOK_URL="${2:-}"
BOOK_BASEURL="${3:-}"

[ ! -e "$SRC" ] || { echo "error: $SRC already exists; give a fresh directory" >&2; exit 1; }
mkdir -p "$SRC"
cp docs/book/*.md "$SRC"/
# Figures. The book's diagrams are .svg FILES referenced with ![](),
# not inline <svg> and not mermaid, because the book is read in two
# places that disagree: GitHub's markdown sanitizer strips inline
# <svg>, and Pages has no mermaid plugin (only jekyll-relative-links
# is enabled below), so a ```mermaid fence renders here as a code
# block. A referenced .svg is the one form that renders in both.
# Without this copy the figures 404 on Pages while looking fine on
# GitHub — which is the failure mode that would ship unnoticed.
if compgen -G "docs/book/*.svg" > /dev/null; then
  cp docs/book/*.svg "$SRC"/
fi
# The chapter table is the front page.
mv "$SRC"/README.md "$SRC"/index.md
cat > "$SRC"/_config.yml <<'CFG'
title: An inference primer for Go engineers
description: How a language model actually runs, for someone who knows Go and not machine learning.
theme: jekyll-theme-primer
# jekyll-relative-links ships enabled in the github-pages gem and rewrites the
# README's ./NN-chapter.md links to their built .html permalinks. Without it,
# every chapter link on the index page 404s.
plugins:
  - jekyll-relative-links
relative_links:
  enabled: true
  collections: false
CFG
if [ -n "$BOOK_URL" ]; then
  printf 'url: %s\nbaseurl: %s\n' "$BOOK_URL" "$BOOK_BASEURL" >> "$SRC"/_config.yml
fi
echo "staged $(ls "$SRC"/*.md | wc -l) markdown files:"
ls "$SRC"

set -euo pipefail
mkdir -p "$SRC"/_includes
# GUARD: the include must contain no Liquid TAGS ({%...%}), only output ({{...}}).
# Liquid is evaluated before HTML comments mean anything, so a tag written inside
# <!-- --> still runs — and because this file IS head-custom.html, a comment naming
# its own include tag made it include itself. That shipped: the build died with
# "stack level too deep ... 10124 levels" blaming the THEME's layout. One grep is
# cheaper than re-deriving that from a stack trace.
if grep -q '{%' docs/book-search/head-custom.html; then
  echo "::error::head-custom.html contains a Liquid tag ({%...%}). Even inside an HTML"
  echo "::error::comment it is executed; a self-include here recurses until the build dies."
  grep -n '{%' docs/book-search/head-custom.html
  exit 1
fi
# GUARD 2: exactly one HTML comment open and one close in the whole file.
#
# The guard above stops the build from DYING. This one stops it from succeeding while
# rendering garbage, which is worse because nothing reports it. The file's leading
# comment explained the Liquid trap by writing an empty HTML comment inline to
# illustrate "inside a comment" — and that inline close ENDED the real comment there,
# so seventeen lines of prose about Liquid escaping rendered as visible text at the
# top of every page of the book. It deployed green.
#
# Counting opens against closes would NOT have caught it: that file had two of each,
# perfectly balanced. The invariant that actually holds for this file is that it
# carries exactly ONE comment, so any second delimiter of either kind is the bug.
opens=$(grep -o -- '<!--' docs/book-search/head-custom.html | wc -l | tr -d ' ')
closes=$(grep -o -- '-->' docs/book-search/head-custom.html | wc -l | tr -d ' ')
if [ "$opens" != "1" ] || [ "$closes" != "1" ]; then
  echo "::error::head-custom.html must contain exactly one HTML comment: found $opens open(s), $closes close(s)."
  echo "::error::A second delimiter closes the leading comment early and its prose renders on every page."
  grep -n -- '<!--\|-->' docs/book-search/head-custom.html
  exit 1
fi
cp docs/book-search/head-custom.html "$SRC"/_includes/head-custom.html
python3 scripts/build_search_index.py "$SRC" "$SRC"/search-index.json
# Non-vacuity: an empty or trivially small index would still deploy and would still
# render a search box that finds nothing, which looks like a working feature.
test "$(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$SRC/search-index.json")" -ge 50
