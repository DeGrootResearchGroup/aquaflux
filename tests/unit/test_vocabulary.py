"""The package's vocabulary: which names mean what, and where each may be defined.

A settings field is read straight from a case file, so its name is a user-facing key, and a parameter
name is how a reader tells what a recorded setting meant. Several words in this package came to name
two different quantities: ``relaxation`` named both an under-relaxation factor in ``(0, 1]``, which
damps *less* as it grows, and the pseudo-transient shift strength ``beta``, which damps *more*;
``rtol`` named both a residual tolerance and a tolerance on the state increment, with different
defaults; ``omega`` named both the turbulence model's specific dissipation rate and a smoother's
damping weight, and one case file set both a few lines apart. Each such name was introduced by a
change that looked right on its own, because nothing compared it with the other places the word was
already in use.

These tests make reusing one of those words a decision that is written down:

* :func:`test_each_overloaded_word_is_defined_only_where_it_has_been_reviewed` -- every module that
  defines a parameter, field, function or method with one of :data:`VOCABULARY`'s words must be listed under that word,
  with the meaning it gives the word there. A module not listed fails, and so does a listed module
  that no longer defines the word, so the list cannot go stale as renames land.
* :func:`test_no_retired_name_is_defined_again` -- a name retired because it meant something other
  than what it said may not be defined again anywhere in the package.

The scan reads function, method and property names, their parameters (lambdas included) and class-level
annotated fields (``equinox`` modules, dataclasses, named tuples) in ``aquaflux/``, by parsing rather
than by matching text.
"""

from __future__ import annotations

import ast
import functools
import pathlib
from dataclasses import dataclass

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "aquaflux"


@dataclass(frozen=True)
class Word:
    """One overloaded word: the meaning it should have, and every module that defines it.

    Attributes
    ----------
    meaning : str
        The single meaning the word is being converged on.
    modules : dict of str to str
        Each module (relative to ``aquaflux/``) that defines a name of this spelling,
        mapped to the meaning it has there. A module using a different meaning says so, and names
        the issue that will rename it.
    """

    meaning: str
    modules: dict[str, str]


SHIFT = "the pseudo-transient shift strength"

#: Overloaded words. Adding a module under a word is the place to ask whether the new use means the
#: same thing; if it does not, use another name rather than listing it.
VOCABULARY: dict[str, Word] = {
    "relaxation": Word(
        meaning="an under-relaxation factor in (0, 1]; larger damps less",
        modules={
            "case/solver.py": "the segregated loop's under-relaxation of k and omega",
            "schemes/gradient.py": "the damping of a Richardson sweep",
            "turbulence/driver.py": "the segregated loop's under-relaxation of k and omega",
        },
    ),
    "beta": Word(
        meaning=SHIFT + "; larger damps more",
        modules={
            "case/run.py": SHIFT,
            "precondition/materialized_session.py": SHIFT,
            "solve/march_history.py": SHIFT,
            "solve/march_log.py": SHIFT,
            "solve/relaxation.py": SHIFT,
            "solve/retry.py": SHIFT,
            "solve/step_control.py": SHIFT,
            "solve/strategy.py": SHIFT,
            "turbulence/coupled.py": SHIFT,
        },
    ),
    "omega": Word(
        meaning="the turbulence model's specific dissipation rate",
        modules={
            "case/boundaries.py": "the specific dissipation rate",
            "turbulence/coupled.py": "the specific dissipation rate",
            "turbulence/driver.py": "the specific dissipation rate",
            "turbulence/initialization.py": "the specific dissipation rate",
            "turbulence/scales.py": "the specific dissipation rate",
            "turbulence/sources.py": "the specific dissipation rate",
            "turbulence/sst.py": "the specific dissipation rate",
            "turbulence/transport.py": "the specific dissipation rate",
        },
    ),
    "mu": Word(
        meaning="dynamic viscosity",
        modules={
            "flow/rhie_chow.py": "dynamic viscosity",
        },
    ),
    "viscosity": Word(
        meaning="the molecular dynamic viscosity of the fluid",
        modules={},
    ),
    "rtol": Word(
        meaning="a relative tolerance on a residual norm",
        modules={
            "case/solver.py": "a relative tolerance on a residual norm",
            "schemes/gradient.py": "a relative tolerance on a residual norm",
            "solve/convergence.py": "a relative tolerance on a residual norm",
            "solve/linear.py": "a relative tolerance on a residual norm",
            "solve/linear_solver_spec.py": "a relative tolerance on a residual norm",
            "solve/march.py": "a relative tolerance on a residual norm",
            "solve/shifted_step.py": "a relative tolerance on a residual norm",
            "solve/strategy.py": "a relative tolerance on a residual norm",
        },
    ),
    "max_steps": Word(
        meaning="the most outer iterations of the solve it caps",
        modules={
            "case/solver.py": "the most outer iterations of a solve",
            "flow/march.py": "the most outer iterations of a solve",
            "triangle_grid_walk.py": "the most cells a ray may cross",
            "solve/driver.py": "the most outer iterations of a solve",
            "solve/implicit.py": "the most outer iterations of a solve",
            "solve/march.py": "the most outer iterations of a solve",
            "turbulence/coupled.py": "the most outer iterations of a solve",
        },
    ),
}

#: Names retired because they did not say what they meant, each with what replaced it.
RETIRED: dict[str, str] = {
    "cycles_limit": "RetryPolicy.max_retries: it counted redone steps, not cycles",
    "row_relaxation": "ShiftTerm.row_shift_scale: it scales the shift strength, which is not a relaxation",
    "pressure_omega": "pressure_smoother_weight: omega is the turbulence model's dissipation rate",
    "omega_smooth": "prolongation_weight: omega is the turbulence model's dissipation rate",
    "v_cycles": "cycles: the name every other multigrid value uses for V-cycles per application",
    "n_group_fields": "n_fields: the field count of the block the inverse is fitted to",
    "f_iters": "f_sweeps: smoother sweeps are counted in sweeps everywhere else",
    "c_iters": "c_sweeps: smoother sweeps are counted in sweeps everywhere else",
    "explicit_limiter": "explicit_production_limiter: the name SSTTurbulence gives the same setting",
    "with_traced_split": "with_traced_form: the traced inverse it swaps in need not be a field split",
}


@functools.cache
def _definitions() -> dict[str, frozenset[str]]:
    """Every module defining each scanned name, keyed by name."""
    names = set(VOCABULARY) | set(RETIRED)
    found: dict[str, set[str]] = {name: set() for name in names}
    for path in sorted(PACKAGE.rglob("*.py")):
        module = path.relative_to(PACKAGE).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                arguments = node.args
                defined = [
                    a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                ]
                if not isinstance(node, ast.Lambda):
                    defined.append(node.name)
            elif isinstance(node, ast.ClassDef):
                defined = [
                    statement.target.id
                    for statement in node.body
                    if isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                ]
            else:
                continue
            for name in names.intersection(defined):
                found[name].add(module)
    return {name: frozenset(modules) for name, modules in found.items()}


def test_each_overloaded_word_is_defined_only_where_it_has_been_reviewed():
    problems = []
    for name, word in VOCABULARY.items():
        defined = _definitions()[name]
        listed = set(word.modules)
        for module in sorted(defined - listed):
            problems.append(
                f"{module} defines `{name}`, which in this package means {word.meaning}. If it means "
                f"that here, list the module under VOCABULARY[{name!r}]; if not, choose another name."
            )
        for module in sorted(listed - defined):
            problems.append(
                f"VOCABULARY[{name!r}] lists {module}, which no longer defines `{name}`: remove it."
            )
    assert not problems, "\n".join(problems)


def test_no_retired_name_is_defined_again():
    problems = [
        f"{module} defines `{name}`, retired in favour of {replacement}."
        for name, replacement in RETIRED.items()
        for module in sorted(_definitions()[name])
    ]
    assert not problems, "\n".join(problems)


def test_the_scan_finds_parameters_fields_lambda_arguments_and_method_names(tmp_path, monkeypatch):
    """The scan must see each kind of definition, or a reuse of a word would pass unnoticed."""
    package = tmp_path / "aquaflux"
    (package / "sub").mkdir(parents=True)
    (package / "sub" / "a.py").write_text("def f(*, relaxation):\n    return relaxation\n")
    (package / "sub" / "b.py").write_text("class C:\n    relaxation: float = 1.0\n")
    (package / "c.py").write_text("g = lambda relaxation: relaxation\n")
    (package / "d.py").write_text("relaxation = 1.0\nprint(relaxation)\n")
    (package / "e.py").write_text(
        "class D:\n    @property\n    def relaxation(self):\n        return 1.0\n"
    )
    monkeypatch.setitem(globals(), "PACKAGE", package)
    _definitions.cache_clear()
    try:
        assert _definitions()["relaxation"] == {"sub/a.py", "sub/b.py", "c.py", "e.py"}
    finally:
        _definitions.cache_clear()
