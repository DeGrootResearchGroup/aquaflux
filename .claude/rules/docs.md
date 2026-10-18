---
paths:
  - "docs/**"
  - "tools/build_docs.sh"
  - ".readthedocs.yaml"
---

# docs — the Sphinx site

The documentation rules, moved verbatim from the root `CLAUDE.md` so they load only while the site
itself is being edited. The root file keeps the build command and the shipped-surface rule; a
documented docstring outside `docs/` does not load this file, so the root tells whoever edits one
to read it.

User-facing docs live in `docs/` as a **Sphinx site in MyST Markdown** (`pydata-sphinx-theme`, autodoc +
napoleon), with an API page (`docs/api.md`) **generated at build time** from each documented
subpackage's `__all__` (`conf.py` `_write_api_page` / `PUBLIC_SUBPACKAGES`). Read the Docs builds it with
`fail_on_warning: true`, so every cross-reference must resolve and every page sit in a toctree. **Build
locally with `tools/build_docs.sh`** — the same `sphinx-build -b html -W docs docs/_build/html` Read the
Docs and CI run, after deleting the gitignored build artifacts, in a cached environment
(`~/.cache/aquaflux/build-venv`) because a PEP-668 system Python refuses `pip install -e ".[docs]"`.
**CI builds the docs on every PR** (the required `fast gate` depends on it); still run
`tools/build_docs.sh` when you touch a documented docstring, a `docs/` page, or a cross-reference.

**Publishing a subpackage means publishing its whole `__all__`** — the export list *is* the editorial
decision; to keep a name off the site, take it out of `__all__`. Page grouping is `SUBPACKAGE_GROUPS`,
keyed on the module each name is **defined** in; extend it when a subpackage grows a module (`-W` will not
tell you). Three traps:

- **A multi-name parameter entry must sit on ONE line** (`beta0, exponent, beta_floor`); a wrapped name
  line, or prose after the last entry of a `Parameters` section, gives the `Field list ends without a
  blank line` / `Unexpected indentation` / `Block quote ends without a blank line` triple. Put such prose
  above `Parameters` or in its own section.
- **A field whose default is a plain function is documented by that function's docstring**
  (`residual_norm = field(default=jnp.linalg.norm)`); `conf.py`'s `_skip_borrowed_member_doc` drops class
  members that are routines defined outside `aquaflux`; document such fields in the class's `Attributes`.
- **Delete `docs/generated/`, `docs/api.md` and `docs/_build/` between probe builds**
  (`rm -rf docs/generated docs/api.md docs/_build`); leftover stubs make a run clean only about the
  previous set of pages.

**A Sphinx extension enabled in `conf.py` must be matched by a dependency in the `docs` extra** (MyST
`linkify` needs `myst-parser[linkify]`; without it *every* page fails). Verify in a **clean** environment.

The docs are the **shipped surface**, so the Comment Convention and the Claude-Facing-File Reference Ban
apply: no references to the precursor codebases, the Claude-facing files (`CLAUDE.md` / `.claude/`), the
internal design notes, or the author's own papers, and no roadmap/build-stage labels ("Milestone 0",
"BUILT"). The annotated `package_structure.md` file tree is internal, stays in `docs/`, and is listed in
`conf.py` `exclude_patterns` so it never reaches the built site.
