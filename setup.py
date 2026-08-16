"""Build the compiled extensions. Everything else about packaging lives in ``pyproject.toml``.

This file exists for one reason: ``setuptools`` cannot declare a Cython extension declaratively, so the
``cythonize`` call needs a hook. Nothing configurational belongs here — adding metadata to this file
instead of ``pyproject.toml`` is how a project ends up with two places to look.

**Why the package carries compiled code at all.** The zero-fill incomplete factorization in
``aquaflux/solve/_ilu0.pyx`` eliminates a row at a time against every row above it, so it is sequential
by nature and cannot be expressed as array operations. It is also the smoother the coupled
preconditioner's CPU path depends on, applied several times per multigrid level per cycle, so a Python
loop over some tens of millions of nonzeros is not a slow version of it — it is not a usable version of
it. The alternative considered and rejected was a drop-tolerance factorization from a library, which is
a different algorithm: it decides what to keep by value, and on this saddle it returns either a singular
factor or one whose entries reach 1e+23.
"""

from __future__ import annotations

import numpy as np
from setuptools import setup

try:
    from Cython.Build import cythonize
except ImportError as error:  # pragma: no cover - the build backend installs it
    raise SystemExit(
        "aquaflux needs Cython to build its compiled extensions. It is declared in the build "
        "requirements, so a normal `pip install` supplies it; a build run against an environment "
        "assembled by hand may not."
    ) from error

setup(
    ext_modules=cythonize(
        ["aquaflux/solve/_ilu0.pyx"],
        # The generated C is not committed, so a version mismatch cannot leave a stale one behind.
        language_level="3",
    ),
    include_dirs=[np.get_include()],
)
