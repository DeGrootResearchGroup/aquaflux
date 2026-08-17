#!/usr/bin/env bash
#
# Build the compiled zero-fill incomplete factorization in place, in THIS checkout.
#
# Why this exists rather than "just run pip install -e .": the built extension is a gitignored
# artifact, so it belongs to a checkout rather than to a branch. A fresh worktree therefore starts
# without it, and `aquaflux.solve.ilu0` falls back to its pure-Python reference twin SILENTLY -- the
# package imports, every test passes, and every result is numerically identical. Only the cost changes,
# and it changes by orders of magnitude on the one routine the coupled preconditioner's CPU path spends
# its time in. That is the worst shape a regression can have: invisible, and it invalidates exactly the
# measurements a study is run to obtain. A run timed on the fallback is a Python triangular solve being
# compared against somebody else's C one.
#
# It has happened: as of 2026-08-17 no checkout on this machine carried the extension, no run log
# recorded which kernel was live, and wall-clock comparisons had been drawn against a host library's
# compiled factorization on that basis.
#
# The second reason is that `pip install -e .` cannot be relied on here. The build needs Cython and
# setuptools, and a system Python managed by a package manager (Homebrew, most Linux distributions)
# refuses to install them -- PEP 668. So this creates a small build environment ONCE, in the user cache,
# with --system-site-packages so it borrows the already-installed numpy rather than building its own,
# and reuses it for every checkout thereafter. Nothing is installed into the interpreter that runs
# aquaflux; the only thing that lands in the checkout is the .so.
#
# Usage
#   tools/build_ext.sh            build if needed; a no-op when the extension is already live
#   tools/build_ext.sh --force    rebuild even if it is
#
# Exits non-zero only if the build was needed and failed.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/aquaflux"
VENV="$CACHE/build-venv"
PYTHON="${PYTHON:-python3}"

force=0
[ "${1:-}" = "--force" ] && force=1

# Ask the interpreter that will RUN aquaflux, not the build one: a .so built for a different Python is
# not importable by this one, and the question is only ever "is it live for the runtime interpreter".
compiled() {
  ( cd "$ROOT" && "$PYTHON" -c 'from aquaflux.solve.ilu0 import COMPILED; raise SystemExit(0 if COMPILED else 1)' ) 2>/dev/null
}

if [ "$force" -eq 0 ] && compiled; then
  echo "build_ext: the compiled incomplete factorization is already live in $ROOT"
  exit 0
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "build_ext: creating the shared build environment in $VENV"
  # --system-site-packages so numpy comes from the runtime interpreter. That also keeps the headers
  # the extension compiles against the same ones the runtime numpy uses, which a self-contained venv
  # would not guarantee.
  "$PYTHON" -m venv --system-site-packages "$VENV" || {
    echo "build_ext: could not create a build environment at $VENV" >&2
    exit 1
  }
  "$VENV/bin/pip" install --quiet --upgrade setuptools Cython || {
    echo "build_ext: could not install the build requirements (setuptools, Cython)" >&2
    exit 1
  }
fi

echo "build_ext: building aquaflux/solve/_ilu0 in $ROOT"
# `build/` and the .so are gitignored, so this dirties nothing tracked.
( cd "$ROOT" && "$VENV/bin/python" setup.py build_ext --inplace ) >/dev/null 2>"$CACHE/build.log" || {
  echo "build_ext: the build FAILED; see $CACHE/build.log" >&2
  tail -n 15 "$CACHE/build.log" >&2
  exit 1
}

if compiled; then
  echo "build_ext: done -- the compiled kernel is live"
  exit 0
fi

# Built, but the runtime interpreter still will not load it. Almost always a Python-version mismatch
# between $PYTHON and whatever built the .so, which is silent otherwise.
echo "build_ext: the build reported success but the extension is still NOT live for $PYTHON." >&2
echo "build_ext: check that $VENV was created by the same interpreter; delete it and re-run." >&2
exit 1
