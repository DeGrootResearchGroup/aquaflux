"""Sphinx configuration for the aquaflux documentation.

The docs are Markdown (MyST). The API reference is generated at build time from the
public ``__all__`` of each documented subpackage (see ``_write_api_page`` below), so it
stays in lock-step with the curated public surface without a hand-maintained list.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from importlib.metadata import version as _dist_version
from pathlib import Path

# -- Project information ------------------------------------------------------

project = "aquaflux"
author = "Christopher DeGroot"
copyright = "2026, Christopher DeGroot"

try:
    release = _dist_version("aquaflux")
except Exception:  # pragma: no cover - source checkout without an install
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# Subpackages whose ``__all__`` is published in the API reference. Extend as further
# subsystems are documented. Listing one publishes *all* of its exports: there is
# deliberately no second "which names to document" list here, since a hand-maintained
# subset would drift from ``__all__``, which is what generating the page prevents. A
# name that does not belong on the site is a name that does not belong in ``__all__``.
PUBLIC_SUBPACKAGES = ["mesh", "solve", "flow", "turbulence", "transport"]

# How the API reference groups each subpackage, keyed on the module a name is DEFINED in.
# That key is far more stable than the individual names, and it cannot silently lose one: a
# module missing from this table still gets a group of its own headed by its own name, and a
# name defined outside the subpackage falls into "Other". The whole ``__all__`` therefore
# reaches the page either way — this table decides only the order and the headings.
SUBPACKAGE_GROUPS = {
    "mesh": [
        ("The mesh and its geometry", ["mesh", "geometry", "cell", "face", "connectivity"]),
        ("Cell zones and face patches", ["groups"]),
        ("Building and transforming a mesh", ["structured", "collapse", "reorder"]),
        ("Quality and connectivity diagnostics", ["quality", "distance", "graph"]),
    ],
    "solve": [
        ("Nonlinear solve", ["implicit", "newton"]),
        (
            "The pseudo-transient march",
            ["continuation", "march", "step_control", "relaxation", "line_search_growth"],
        ),
        ("Observing and checkpointing a march", ["march_log", "checkpoint", "refresh_timing"]),
        ("Linear solves and residual measures", ["linear", "norm", "shift_basis"]),
        (
            "Preconditioners",
            [
                "host_preconditioner",
                "amg_preconditioner",
                "ilut_preconditioner",
                "lu_preconditioner",
                "field_split",
                "native_inverse",
                "saddle_multigrid",
            ],
        ),
        ("Multigrid hierarchies", ["multigrid", "frozen_operator"]),
        ("Sparse Jacobians", ["sparse_jacobian"]),
    ],
    "flow": [
        ("The momentum-continuity system", ["momentum", "rhie_chow", "source"]),
        ("Boundary conditions", ["boundary"]),
        ("Solving a flow", ["continuation", "mean_velocity"]),
        ("Preconditioners", ["block_preconditioner", "preconditioner"]),
        ("Initialization and flow scales", ["initialization", "scales"]),
    ],
    "transport": [
        ("Scalar transport", ["scalar"]),
    ],
    "turbulence": [
        ("The k-ω SST model", ["sst", "transport", "sources", "strain"]),
        ("Wall treatment and boundary closures", ["boundary"]),
        ("The coupled flow-turbulence solve", ["coupled", "driver", "continuation"]),
        ("Reynolds-number continuation", ["reynolds"]),
        ("Preconditioners", ["preconditioner"]),
        ("Initialization and diagnostics", ["initialization", "diagnostics"]),
    ],
}

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# Every published page is reachable from a toctree; the dev-internal annotated
# file tree is a repo reference, not a user doc, so it is kept out of the built
# site.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "package_structure.md",
]

# -- MyST (Markdown) ----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives
    "deflist",
    "dollarmath",  # $...$ / $$...$$ math
    "fieldlist",
    "linkify",  # bare URLs -> links
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3  # auto-slug headings h1-h3 so cross-page #anchors resolve

# -- autodoc / autosummary ----------------------------------------------------

# Only run *intentional* doctests -- explicit ``.. doctest::`` / ``.. testcode::``
# directives -- not every illustrative ``>>>`` block. Docstring "Examples"
# sections are illustrative (they reference names built earlier in prose) and
# are not meant to execute; testing them wholesale would fail the gate on
# documentation that is correct as documentation.
doctest_test_doctest_blocks = ""

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render a docstring `Attributes` section as inline ``:ivar:`` fields rather than
# separate attribute directives -- otherwise an attribute documented in both the
# `Attributes` section and by autodoc's member scan collides ("duplicate object
# description").
napoleon_use_ivar = True

# -- intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
# Left at Sphinx's default so type references resolve into the inventories above: the
# published solver signatures are dense with `jnp.ndarray` and friends, and each becomes a
# link rather than plain text. This was previously suppressed with `["*"]` to stop missing
# third-party refs failing the `-W` build -- but `nitpicky` is off (the default), so an
# unresolvable reference is silently left as text and cannot fail that gate either way.
# Suppressing it therefore bought nothing and cost every link. Our own names are unaffected:
# a local target always wins, so nothing in `aquaflux` resolves to a third-party page.

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"aquaflux {release}"
html_theme_options = {
    "github_url": "https://github.com/DeGrootResearchGroup/aquaflux",
    "navigation_with_keys": True,
    "show_prev_next": True,
}


# -- Auto-generated API reference ---------------------------------------------


def _grouped_exports(subpackage, module):
    """Split a subpackage's ``__all__`` into ordered, headed groups for the page.

    Each name is filed under the module it is *defined* in, and those modules are
    gathered into the reader-facing groups of :data:`SUBPACKAGE_GROUPS`, in that order.
    A module the table does not mention gets a group headed by its own name, and a name
    defined outside the subpackage goes to "Other" -- so grouping only ever reorders the
    exports, it can never drop one.

    Parameters
    ----------
    subpackage : str
        The subpackage's name, without the ``aquaflux.`` prefix.
    module : module
        The imported subpackage, read for its ``__all__``.

    Returns
    -------
    list of (str, list of str)
        ``(heading, names)`` pairs in the order they should appear. Every name in
        ``module.__all__`` appears in exactly one pair.
    """
    prefix = f"aquaflux.{subpackage}."
    by_module = {}
    for name in sorted(module.__all__, key=str.lower):
        home = getattr(getattr(module, name), "__module__", None) or ""
        key = home[len(prefix) :] if home.startswith(prefix) else None
        by_module.setdefault(key, []).append(name)

    groups, claimed = [], set()
    for heading, modules in SUBPACKAGE_GROUPS.get(subpackage, ()):
        claimed.update(modules)
        names = sorted((n for m in modules for n in by_module.get(m, ())), key=str.lower)
        if names:
            groups.append((heading, names))
    for key in sorted(k for k in by_module if k is not None and k not in claimed):
        groups.append((key.replace("_", " ").capitalize(), by_module[key]))
    if None in by_module:
        groups.append(("Other", by_module[None]))
    return groups


def _write_api_page(app):
    """Generate ``api.md`` from each documented subpackage's ``__all__``.

    For every subpackage in :data:`PUBLIC_SUBPACKAGES`, its exports are gathered into
    ``autosummary`` tables by :func:`_grouped_exports`; ``autosummary_generate`` then
    emits a stub page per object under ``generated/``.
    """
    lines = [
        "# API reference",
        "",
        "The complete curated public API of each documented subpackage, generated from "
        "its ``__all__``.",
        "",
    ]

    for subpackage in PUBLIC_SUBPACKAGES:
        module = import_module(f"aquaflux.{subpackage}")

        def table(heading, names, subpackage=subpackage):
            if not names:
                return []
            body = "\n".join(f"   aquaflux.{subpackage}.{n}" for n in names)
            return [
                f"### {heading}",
                "",
                "```{eval-rst}",
                ".. autosummary::",
                "   :toctree: generated",
                "   :nosignatures:",
                "",
                body,
                "```",
                "",
            ]

        lines += [f"## `aquaflux.{subpackage}`", ""]
        for heading, names in _grouped_exports(subpackage, module):
            lines += table(heading, names)

    (Path(app.srcdir) / "api.md").write_text("\n".join(lines) + "\n")


def _skip_borrowed_member_doc(app, what, name, obj, skip, options):
    """Skip a class member documented only by a *third-party* callable it happens to hold.

    autodoc omits undocumented attributes by default, which is what keeps the solvers'
    configuration fields out of the reference -- each is described in its class's ``Attributes``
    section instead. A field whose default value is a plain function escapes that rule: the class
    attribute *is* that function, so autodoc finds the function's own docstring and documents the
    field with it. ``residual_norm``, which defaults to the Euclidean norm, would render the array
    norm routine's documentation -- prose about a different subject, written to another project's
    conventions, and not necessarily valid reStructuredText here.

    Such a member is undocumented in every sense that matters, so it is dropped like any other.
    The test is where the callable was defined: a method written in this package documents itself,
    while a callable from elsewhere can only ever contribute borrowed prose. Returning ``None``
    leaves every other member to autodoc's own decision.
    """
    del app, name, options
    if what != "class" or skip or not inspect.isroutine(obj):
        return None
    defining_module = getattr(obj, "__module__", None) or ""
    return True if not defining_module.startswith("aquaflux") else None


def setup(app):
    app.connect("config-inited", lambda app, config: _write_api_page(app))
    app.connect("autodoc-skip-member", _skip_borrowed_member_doc)
