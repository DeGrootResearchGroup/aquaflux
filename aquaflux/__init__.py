"""aquaflux — differentiable unstructured finite-volume flow solver in JAX.

A bespoke, fully differentiable, cell-centred FVM flow solver built to couple with
``aquakin`` for water and environmental engineering reactors.

Importing ``aquaflux`` enables JAX 64-bit mode process-wide (see below); this is a
documented, intentional side effect required for finite-volume and stiff-coupling
accuracy.
"""

from __future__ import annotations

# --- JAX x64 enablement (must run before any submodule builds JAX state) --------
# Finite-volume assembly and stiff reactive coupling require 64-bit floats. This is
# global, process-wide JAX configuration, so it is a documented side effect of
# ``import aquaflux``. It is set here, at the top of the package init, before any
# submodule import that would construct JAX arrays/tracers. Do not remove it.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.0"

__all__ = ["__version__"]

# NOTE: the public API surface is intentionally empty at this pre-code scaffold
# stage. Export symbols here as subsystems land (mesh, discretization, solve),
# keeping this file's import order compatible with the x64 enablement above.
