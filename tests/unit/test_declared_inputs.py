"""Every shipped residual term's declared inputs, checked against what its evaluation reads.

:class:`~aquaflux.discretization.DeclaredInputs` lets a term say which named properties it reads
(``requires``) and whether its answer needs a reconstructed gradient (``uses_gradient``), and the
assemblers refuse a configuration that cannot supply them. A declaration nothing falsifies drifts from
the reads it describes, so this module evaluates each concrete term in the package and checks the
declaration both ways:

* against **exactly** its declared properties it evaluates to a finite answer -- a property read but
  not declared is missing from the mapping and raises;
* with each declared property **poisoned by NaN** the answer is not finite -- a declared property the
  term never reads would leave it finite. NaN rather than zero, because a zero is a value many terms
  read without complaint, and reading a silent zero is the failure the declaration exists to prevent;
* on an orthogonal grid, ``uses_gradient()`` holds exactly when replacing the gradient by zero
  changes the answer -- the meaning the contract gives it, which is not "reads the gradient at all"
  (a diffusion flux reads it for a non-orthogonal correction that vanishes on such a grid).

Coverage is declared rather than discovered: every concrete subclass of ``DeclaredInputs`` defined in
the package must have a case in :data:`CASES`, so a new term cannot ship without its declaration
being checked.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable

import aquaflux
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.context import FieldContext, MeshContext
from aquaflux.discretization import (
    AdvectionFlux,
    AdvectionScheme,
    DeclaredInputs,
    DiffusionFlux,
    FaceFluxOperator,
    FirstOrderUpwind,
    LimitedUpwind,
    TransientTerm,
    VolumeSource,
)
from aquaflux.flow import MomentumSource, PressureForce, UniformBodyForce, VelocityFields
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes import VenkatakrishnanLimiter
from aquaflux.turbulence import (
    KDestruction,
    KProduction,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
    SSTModel,
)

# An orthogonal grid: `uses_gradient` is defined by what a zero gradient changes on one.
MESH = structured_grid_2d(4, 3)
GEOMETRY = MESH.geometry()
N_CELLS, N_FACES, DIM = MESH.n_cells, MESH.n_faces, MESH.dim

_rng = np.random.default_rng(362)


def _positive(*shape: int) -> jnp.ndarray:
    return jnp.asarray(_rng.uniform(0.5, 2.0, size=shape))


def _signed(*shape: int) -> jnp.ndarray:
    return jnp.asarray(_rng.uniform(-1.0, 1.0, size=shape))


FIELD = _positive(N_CELLS)
BOUNDARY_VALUES = _positive(N_FACES)
# Mixed signs, so every face reconstruction sees both an outflow and an inflow side somewhere.
MASS_FLUX = _signed(N_FACES)
CELL = _positive(N_CELLS)
SCALAR_GRADIENT = _signed(N_CELLS, DIM)
VELOCITY = _signed(N_CELLS, DIM)
BOUNDARY_VELOCITY = _signed(N_FACES, DIM)
VELOCITY_GRADIENT = _signed(N_CELLS, DIM, DIM)


#: One configured instance of every concrete term in the package. Where a term's declaration depends
#: on its configuration, the configuration chosen is the one that exercises it: a diffusion flux on a
#: non-default coefficient name, a limited upwind scheme with its limiter set.
CASES: dict[type, Callable[[], DeclaredInputs]] = {
    DiffusionFlux: lambda: DiffusionFlux(coefficient="conductivity"),
    AdvectionFlux: lambda: AdvectionFlux(mass_flux=MASS_FLUX, scheme=LimitedUpwind()),
    PressureForce: lambda: PressureForce(face_pressure=_positive(N_FACES), component=1),
    FirstOrderUpwind: FirstOrderUpwind,
    LimitedUpwind: lambda: LimitedUpwind(limiter=VenkatakrishnanLimiter(softening=0.05, scale=1.0)),
    KProduction: lambda: KProduction(
        nu_t=CELL, strain_rate=_positive(N_CELLS), omega=_positive(N_CELLS), model=SSTModel()
    ),
    KDestruction: lambda: KDestruction(omega=CELL, model=SSTModel()),
    OmegaProduction: lambda: OmegaProduction(
        strain_rate=CELL,
        nu_t=_positive(N_CELLS),
        k=_positive(N_CELLS),
        omega=_positive(N_CELLS),
        f1=jnp.full(N_CELLS, 0.5),
        model=SSTModel(),
    ),
    OmegaDestruction: lambda: OmegaDestruction(f1=jnp.full(N_CELLS, 0.5), model=SSTModel()),
    OmegaCrossDiffusion: lambda: OmegaCrossDiffusion(
        omega=CELL,
        grad_k=_signed(N_CELLS, DIM),
        grad_omega=_signed(N_CELLS, DIM),
        f1=jnp.full(N_CELLS, 0.5),
        model=SSTModel(),
    ),
    TransientTerm: TransientTerm,
    UniformBodyForce: lambda: UniformBodyForce(force=jnp.array([1.0, -2.0])),
}


def _evaluate(term: DeclaredInputs, properties: dict, *, gradient: bool) -> jnp.ndarray:
    """Everything ``term`` computes, flattened, given ``properties`` and a real or a zero gradient.

    Each family is evaluated through its own public members, against the input its assembler hands
    it. A momentum source contributes all three of its members, since each one reads properties.
    """
    if isinstance(term, MomentumSource):
        tensor = VELOCITY_GRADIENT if gradient else jnp.zeros_like(VELOCITY_GRADIENT)
        fields = VelocityFields(VELOCITY, BOUNDARY_VELOCITY, tensor)
        parts = [
            term.source(fields, GEOMETRY, properties),
            term.diagonal(VELOCITY, GEOMETRY, properties),
        ]
        face_force = term.face_force(GEOMETRY, properties)
        if face_force is not None:
            parts.append(face_force)
        return jnp.concatenate([jnp.ravel(part) for part in parts])
    if isinstance(term, TransientTerm):
        # It is handed no context: neither properties nor a gradient can reach it.
        return term.residual(FIELD, 0.9 * FIELD, 0.8 * FIELD, 0.1, False, GEOMETRY.cell.volume)
    context = FieldContext(
        mesh=MeshContext(face_cells=MESH.face_cells, geometry=GEOMETRY, properties=properties),
        boundary_values=BOUNDARY_VALUES,
        gradient=SCALAR_GRADIENT if gradient else jnp.zeros_like(SCALAR_GRADIENT),
    )
    if isinstance(term, FaceFluxOperator):
        return term.face_flux(FIELD, context)
    if isinstance(term, VolumeSource):
        return term.source(FIELD, context)
    if isinstance(term, AdvectionScheme):
        return term.face_value(FIELD, context, MASS_FLUX)
    raise TypeError(f"no evaluation for the term family of {type(term).__name__}")


def _declared(term: DeclaredInputs) -> dict:
    return {name: _positive(N_CELLS) for name in term.requires()}


def _concrete_terms() -> set[type]:
    """Every concrete ``DeclaredInputs`` subclass defined in the package, after importing all of it."""
    for module in pkgutil.walk_packages(aquaflux.__path__, "aquaflux."):
        if not module.name.endswith("__main__"):
            importlib.import_module(module.name)
    found, pending = set(), [DeclaredInputs]
    while pending:
        for sub in pending.pop().__subclasses__():
            pending.append(sub)
            if sub.__module__.startswith("aquaflux.") and not inspect.isabstract(sub):
                found.add(sub)
    return found


def test_every_concrete_term_in_the_package_has_a_case() -> None:
    """A term with no case would ship an unchecked declaration; a case with no term is stale."""
    found = _concrete_terms()
    missing = sorted(t.__qualname__ for t in found - CASES.keys())
    stale = sorted(t.__qualname__ for t in CASES.keys() - found)
    assert not missing, f"add a case to CASES for {missing}"
    assert not stale, f"CASES names classes that are not concrete terms: {stale}"


@pytest.fixture(params=list(CASES), ids=lambda cls: cls.__name__)
def term(request) -> DeclaredInputs:
    built = CASES[request.param]()
    assert type(built) is request.param
    return built


def test_a_term_evaluates_against_exactly_its_declared_properties(term) -> None:
    result = _evaluate(term, _declared(term), gradient=True)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_every_property_a_term_declares_is_read(term) -> None:
    for name in term.requires():
        poisoned = {**_declared(term), name: jnp.full(N_CELLS, jnp.nan)}
        result = _evaluate(term, poisoned, gradient=True)
        assert not bool(jnp.all(jnp.isfinite(result))), f"{name!r} is declared but never read"


def test_uses_gradient_says_whether_a_zero_gradient_changes_the_answer(term) -> None:
    properties = _declared(term)
    with_gradient = _evaluate(term, properties, gradient=True)
    without = _evaluate(term, properties, gradient=False)
    changes = not bool(jnp.allclose(with_gradient, without, rtol=0.0, atol=1e-12))
    assert term.uses_gradient() == changes


def test_the_cases_exercise_both_answers_of_each_declaration() -> None:
    """Guards the checks above against becoming vacuous: at least one shipped term reads a property
    and at least one needs the gradient, so neither direction is checked only on empty declarations."""
    terms = [build() for build in CASES.values()]
    assert any(t.requires() for t in terms)
    assert any(t.uses_gradient() for t in terms)
    assert any(not t.uses_gradient() for t in terms)
