# Steady-state solving

Many flow problems only ever want the steady answer: the velocity and pressure fields a
reactor settles into, not the path it took to get there. `aquaflux` solves for that state
**directly** — it finds the root of the steady residual with Newton's method — rather than
integrating a transient forward until it stops changing.

Solving directly is attractive because a transient march has to resolve every time scale in
the problem on the way to a state where none of them matter. It is also demanding: Newton
only converges from a good enough starting guess, and a poor one diverges. This page covers
the three things that make a direct steady solve work here — an exact Jacobian, a
globalization strategy that gets Newton into the basin, and the adjoint that makes the
converged state differentiable — and how to assemble them.

## The residual is the model

Every model in `aquaflux` reduces to a **residual**: a function that takes the state (the
unknown fields, flattened into one vector) and returns how badly each cell's conservation
equation is violated. A steady solution is a state where that residual is zero.

For the coupled pressure–velocity system the residual is assembled by a
`MomentumContinuity`, built over a mesh from the fluid properties, a gradient scheme, and the
boundary conditions:

```python
import aquaflux  # noqa: F401  (enables 64-bit mode)
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss

mesh = structured_grid_2d(32, 32, lx=1.0, ly=1.0, named_boundaries=True)
geometry = mesh.geometry()

boundary = BoundaryConditions(
    {
        "top": MovingWall(velocity=(1.0, 0.0)),
        "bottom": NoSlipWall(),
        "left": NoSlipWall(),
        "right": NoSlipWall(),
    }
)

cavity = MomentumContinuity.build(
    mesh,
    geometry,
    PropertyModel({"viscosity": Constant(0.01), "density": Constant(1.0)}),
    CompactGreenGauss(),
    boundary,
    advection_scheme=FirstOrderUpwind(),
    pressure_pin=0,        # a closed domain fixes the pressure datum at one cell
)

state = cavity.initial_state()          # the flat [velocity..., pressure] vector
residual = cavity.residual(state)       # zero at the steady solution
```

This is the lid-driven cavity: a square of fluid with three stationary walls and a lid sliding
across the top. Because momentum is advected, the residual is **nonlinear** — the mass flux
and the advected velocity both depend on the velocity — so finding its root takes several
Newton steps rather than one.

## Newton with an exact Jacobian

Newton's method needs the Jacobian of the residual. `aquaflux` never writes one down: the
residual is JAX code, so the Jacobian comes from automatic differentiation, and it is the
exact derivative of the discretization that is actually solved. Nothing is lagged, frozen, or
approximated behind the solver's back.

That matters most where a hand-derived linearization is usually incomplete. On a
**non-orthogonal** mesh, the diffusion flux carries a correction term for the skewness between
neighbouring cells; codes that linearize by hand normally lag that correction and recover it
over several outer sweeps. Here it is differentiated with everything else, so a linear problem
on a skewed mesh still converges in a **single** Newton step.

For a residual that is genuinely linear — Stokes flow (build the assembler with no
`advection_scheme`), or a scalar diffusion problem — one correction is the whole answer, and
`newton_step` gives it directly:

```python
from aquaflux.solve import newton_step

state = newton_step(stokes.residual, stokes.initial_state())   # exact for a linear residual
```

For a nonlinear residual you want the full solver, which iterates to a convergence test
rather than a fixed number of steps, and carries the adjoint described below.

## Globalization: reaching the basin

An undamped Newton step is only reliable near the root. Starting from a uniform field at any
appreciable Reynolds number, the full step overshoots — often catastrophically — and the
iteration diverges. **Globalization** is what reshapes the early part of the path so the
iteration stays somewhere Newton can work from.

`aquaflux` expresses this as one injected strategy, the solver's `forward_step`. Two are
built in, and both share a property worth stating up front:

```{important}
Every globalization strategy vanishes at the fixed point. Whatever damping or shift it applies
on the way in is zero once the residual is zero, so the **converged state does not depend on
which strategy produced it** — and neither does its gradient. Globalization buys you a path,
never a different answer.
```

### Line search — the default

`DampedNewtonStep` computes the full Newton correction and then backtracks: it takes the
longest step from `1, 1/2, 1/4, …` that actually reduces the residual. Near the root the full
step already reduces it, so the search costs one residual evaluation and the iteration is
undamped — you keep Newton's fast terminal convergence and pay for the search only when you
need it.

```python
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver

solver = ImplicitNewtonSolver(
    max_steps=30,
    forward_step=DampedNewtonStep(line_search=10),
)
state = solver.solve(lambda s, a: a.residual(s), cavity.initial_state(), cavity)
```

The residual is passed as `residual_fn(state, params)` with the parameters explicit — that
second argument is what the adjoint returns sensitivities for, so pass the assembler there
rather than capturing it in the closure.

### Pseudo-transient continuation — for convection-dominated flow

A line search has limits. On a strongly convective flow it can only shorten a direction that
is pointing the wrong way, and past a certain Reynolds number no step length along it helps.
`momentum_continuation` builds the stronger option: it adds a diagonal shift to the Jacobian,
proportional to the momentum equation's own central coefficient, which damps the step in the
manner of an implicit pseudo-time march. The shift ramps down as the residual falls, so the
iteration recovers the exact steady Newton step as it converges.

Here `channel` is an inlet/outlet duct assembler, built exactly as `cavity` was above but with
a `VelocityInlet` and a `PressureOutlet` in place of two of the walls (and no pressure pin — the
outlet sets the datum):

```python
from aquaflux.flow import momentum_continuation

continuation = momentum_continuation(channel, beta0=2.0)
solver = ImplicitNewtonSolver(max_steps=120, forward_step=continuation)
state = solver.solve(lambda s, a: a.residual(s), channel.initial_state(), channel)
```

`beta0` sets the initial damping — larger is more conservative and slower. It is a starting
guess rather than a value that has to be tuned per case: a step that fails to make progress is
automatically re-damped and retried, so choosing `beta0` too small is recovered rather than
fatal.

A good starting field helps both strategies. `potential_flow` builds one by solving a cheap
potential problem for a divergence-free velocity that already respects the geometry and the
inlet:

```python
from aquaflux.flow import potential_flow

state = solver.solve(lambda s, a: a.residual(s), potential_flow(channel), channel)
```

## Preconditioning the linear solve

Each Newton step solves a linear system, and for the coupled pressure–velocity block that
system is a saddle-point problem: it does not respond to a generic Krylov method without help.
`BlockPreconditioner` supplies that help, and the strategy carries it:

```python
from aquaflux.flow import BlockPreconditioner

precond = BlockPreconditioner.build(cavity).factory()
solver = ImplicitNewtonSolver(
    max_steps=30,
    forward_step=DampedNewtonStep(preconditioner=precond),
)
```

Two properties are worth relying on. A preconditioner only accelerates the Krylov iteration —
it never enters the converged state or its gradient, so it can be **frozen** at one operating
point and reused across a whole sweep of solves at different viscosities without biasing any
of them. And because the inner solves need only make Newton progress rather than be exact,
each step's linear solve is taken to a loose tolerance; the outer iteration corrects whatever
is left over. The single adjoint solve is the exception and is taken tightly, because it sets
the accuracy of the gradient directly.

```{warning}
Build the preconditioner **outside** `jax.grad`, from concrete parameter values, as in the
example above. Constructing one inside a differentiated function captures JAX tracers in an
object that is deliberately not differentiated, and the solve then fails with a tracer-leak
error.
```

## Gradients through the converged state

The point of solving in JAX is that the answer is differentiable. It would be ruinous to get
there by taping every Newton iteration — memory would grow with the iteration count, and the
iteration count is data-dependent. `aquaflux` does not: the converged state is defined
implicitly by `R(state, params) = 0`, so the **implicit function theorem** gives its derivative
without reference to how the root was found. The reverse-mode gradient is one transpose linear
solve at the converged state, whose cost is completely independent of how many Newton steps
the forward solve took.

You do not have to ask for any of this. `solver.solve` is differentiable — here, with respect
to the fluid's viscosity, by rebuilding the assembler inside the differentiated function:

```python
import jax
import jax.numpy as jnp

def cavity_at(viscosity):
    """The cavity assembler of the first example, at a given viscosity."""
    return MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(viscosity), "density": Constant(1.0)}),
        CompactGreenGauss(),
        boundary,                      # the BoundaryConditions built above
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )

precond = BlockPreconditioner.build(cavity_at(0.01)).factory()   # built once, outside grad
solver = ImplicitNewtonSolver(max_steps=30, forward_step=DampedNewtonStep(preconditioner=precond))

def mean_speed(viscosity):
    assembler = cavity_at(viscosity)
    state = solver.solve(lambda s, a: a.residual(s), assembler.initial_state(), assembler)
    velocity, _ = assembler.unpack(state)
    return jnp.mean(jnp.abs(velocity[:, 0]))

sensitivity = jax.grad(mean_speed)(0.01)   # d(mean speed) / d(viscosity)
```

The same holds for boundary values, source terms, and the mesh node coordinates — anything the
residual depends on.

```{note}
The steady solve is differentiable in **reverse mode** (`jax.grad`, `jax.vjp`), which is what a
scalar objective over a whole field needs. Forward-mode differentiation (`jax.jacfwd`, `jax.jvp`)
through `ImplicitNewtonSolver` raises; use `newton_step` where a forward-mode derivative through
a linear solve is what you want.
```

## Non-convergence is an error, not a result

The adjoint above is only valid **at a root**. Linearizing at a state that does not solve
`R = 0` still produces a well-posed transpose solve and a perfectly finite gradient — a wrong
one, with nothing to signal that it is wrong.

So the solver refuses to return it. If the iteration exhausts `max_steps` short of tolerance,
or the residual norm becomes non-finite, `solve` raises `equinox.EquinoxRuntimeError` rather
than handing back a state that cannot be trusted. This happens on the `jax.grad` path too, so a
sensitivity can never be built quietly on an unconverged field.

If you hit it, the useful responses in order are: start from a better initial field
(`potential_flow`), switch from the line search to `momentum_continuation`, raise `max_steps`,
and only then loosen `rtol`/`atol`.

## Choosing the pieces

| Situation | Forward step | Notes |
| --- | --- | --- |
| Linear residual (Stokes, scalar diffusion) | `newton_step` | Exact in one call; differentiable in both modes. |
| Nonlinear, moderate Reynolds number | `DampedNewtonStep` | The default. Add a `preconditioner` for any coupled flow. |
| Convection-dominated / high Reynolds number | `momentum_continuation` | Pseudo-transient damping that ramps to zero; pair it with `potential_flow`. |
| Repeated solves at varying viscosity | `reused_flow_solve` | Builds the preconditioned strategy once and reuses the compiled solve across calls. |

The default tolerances (`rtol=1e-10`, `atol=1e-12`) are deliberately tight, since the residual
norm is what the adjoint's validity rests on. `max_steps` defaults to 50, which suits a
line-searched solve; a continuation march legitimately takes more.
