#!/usr/bin/env bash
#
# Build the Sphinx documentation locally, the same strict way CI and Read the Docs build it.
#
# Why this exists rather than "just run pip install -e '.[docs]' && cd docs && make html": the docs
# toolchain (Sphinx, the pydata theme, MyST, sphinx-copybutton) is not part of the runtime dependency
# set, and on this machine `pip install` is refused outright -- the system Python is Homebrew's and is
# externally managed (PEP 668). This exact strict build (`-W`, so any warning is a failure) should be
# run locally whenever a docstring of a documented subpackage, a docs/ page, or a cross-reference
# changes, rather than leaning on CI as the first line of defence -- so "cannot actually run it on this
# machine" is not a state to be in.
#
# This reuses tools/build_ext.sh's fix for the same wall: a small build environment created ONCE in
# the user cache with --system-site-packages, so it borrows the runtime interpreter's already-installed
# numpy/jax/scipy/equinox/lineax/diffrax rather than reinstalling them -- which is exactly what autodoc
# needs to import aquaflux and its dependency graph -- and is reused across checkouts thereafter.
#
# It REUSES build_ext.sh's venv (the shared `build-venv`) rather than creating a sibling one. Both
# scripts exist to route around the identical PEP 668 wall with the identical --system-site-packages
# trick, so keeping them in one environment means there is one build environment to create, locate, and
# blow away when something goes wrong -- not two that can silently drift apart (different pip versions,
# one rebuilt after a Python upgrade and the other stale). The cost is a handful of extra packages
# living alongside the Cython/setuptools build tooling; Sphinx and setuptools do not conflict, and the
# docs toolchain adds only about 100 MB.
#
# aquaflux itself is deliberately NOT pip-installed into that venv. Sphinx is invoked as
# `python -m sphinx`, which (like `python -c`) puts the current directory on sys.path -- run from the
# repo root, that already lets autodoc import aquaflux straight out of this checkout, the same way
# `pytest` does today (aquaflux has never been pip-installed into any interpreter on this machine; see
# tools/build_ext.sh's own header). Installing it here too would mean running `pip install -e .` a
# second time, through a second interpreter, which rebuilds the Cython extension again -- a
# build_ext.sh concern, not this script's -- and could leave a `.so` in the checkout built by a
# DIFFERENT Python than $PYTHON if the two interpreters ever diverge. `docs/conf.py` already handles
# the resulting "no installed distribution" case gracefully (it falls back to a placeholder version);
# its own comment says so.
#
# Usage
#   tools/build_docs.sh    build the docs; exits non-zero on any warning or failure
#
# On success, prints the path to the rendered HTML.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/aquaflux"
VENV="$CACHE/build-venv"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "build_docs: creating the shared build environment in $VENV"
  # --system-site-packages: see the header. This is the same venv tools/build_ext.sh creates, so
  # either script may be the one that creates it first.
  "$PYTHON" -m venv --system-site-packages "$VENV" || {
    echo "build_docs: could not create a build environment at $VENV" >&2
    exit 1
  }
fi

# The docs extra's version pins live in ONE place, pyproject.toml -- read them from there rather than
# hardcoding a second copy here that can silently drift the moment that extra is bumped. tomllib is
# stdlib from Python 3.11; on an older interpreter, fall back to installing the extra directly (which
# is correct but also pip-installs aquaflux itself -- the tradeoff the header above explains, accepted
# here only because this is not the common path).
DOCS_DEPS="$(
  AQUAFLUX_PYPROJECT="$ROOT/pyproject.toml" "$VENV/bin/python" - <<'PY' 2>/dev/null
import os

try:
    import tomllib
except ModuleNotFoundError:
    raise SystemExit(1)

with open(os.environ["AQUAFLUX_PYPROJECT"], "rb") as f:
    data = tomllib.load(f)
print("\n".join(data["project"]["optional-dependencies"]["docs"]))
PY
)"

if [ -z "$DOCS_DEPS" ]; then
  echo "build_docs: $VENV's python has no tomllib; installing aquaflux[docs] directly instead"
  ( cd "$ROOT" && "$VENV/bin/pip" install --quiet --disable-pip-version-check -e ".[docs]" ) || {
    echo "build_docs: could not install the docs toolchain" >&2
    exit 1
  }
else
  echo "build_docs: installing the docs toolchain (pins read from pyproject.toml's [docs] extra)"
  # shellcheck disable=SC2086  # DOCS_DEPS is one PEP 508 requirement string per line, by construction.
  "$VENV/bin/pip" install --quiet --disable-pip-version-check $DOCS_DEPS || {
    echo "build_docs: could not install the docs toolchain" >&2
    exit 1
  }
fi

# docs/generated, docs/api.md and docs/_build are gitignored build artifacts. A stale docs/generated or
# docs/api.md left over from a previous PUBLIC_SUBPACKAGES makes a clean-looking build meaningless -- it
# can be clean about the wrong set of pages.
rm -rf "$ROOT/docs/generated" "$ROOT/docs/api.md" "$ROOT/docs/_build"

echo "build_docs: building $ROOT/docs (warnings are errors)"
if ! ( cd "$ROOT" && "$VENV/bin/python" -m sphinx -b html -W docs docs/_build/html ); then
  echo "build_docs: the build FAILED" >&2
  exit 1
fi

echo "build_docs: done -- open $ROOT/docs/_build/html/index.html"
