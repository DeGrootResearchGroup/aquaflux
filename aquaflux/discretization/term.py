"""The inputs a residual term declares it reads, shared by every family of term.

A residual is assembled from several families of term -- face fluxes, the advection schemes inside
them, volume sources, the transient term, and the momentum sources of the coupled flow. Each is
evaluated against some shared per-evaluation input (a :class:`~aquaflux.context.FieldContext`, or the
flow's kinematic state and its evaluated properties), and what it reads from that input beyond the
field it acts on is not visible from its signature. :class:`DeclaredInputs` is the one contract
through which every family states it, before any residual is evaluated:

* :meth:`~DeclaredInputs.requires` -- the evaluated properties it reads by name;
* :meth:`~DeclaredInputs.uses_gradient` -- whether its answer needs a reconstructed gradient.

An assembler checks both when it is built, so a mistyped property name or a gradient-reading term
with no reconstruction is refused there rather than surfacing inside a traced residual as a bare
``KeyError``, a non-finite value, or a silently less accurate answer. Each family inherits the
contract rather than redeclaring the subset it happened to need, so a term of any family can answer
both questions and an assembler can ask them of every term it holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
    from collections.abc import Iterable


class DeclaredInputs(eqx.Module):
    """Base contract: what a residual term reads beyond the field it acts on.

    Both members default to "nothing", so a term that reads no named property and no gradient
    overrides neither. A declaration is a statement about the term's evaluation, not a hint: a term
    that reads a property it does not declare fails when evaluated against exactly its declared
    properties, and one that declares a property it never reads carries a stale requirement that
    refuses perfectly good property models. Keep each override next to the read it describes.
    """

    def requires(self) -> tuple[str, ...]:
        """Names this term reads from the evaluated properties (default: none).

        Override when the term reads a property by key -- a diffusion flux its coefficient, a drag
        its viscosity -- so that the assembler holding it can check the property model supplies the
        name when it is built.

        Returns
        -------
        tuple of str
            The property names read, in any order.
        """
        return ()

    def uses_gradient(self) -> bool:
        """Whether this term needs a reconstructed gradient (default: ``False``).

        With no gradient scheme an assembler hands its terms a gradient that is exactly zero. For
        most terms that is a graceful degradation: a non-orthogonal correction is *exactly* zero on
        an orthogonal grid, so a diffusion flux stays correct there and does not declare this, even
        though it reads the gradient. A term whose answer depends on the gradient even on an
        orthogonal grid is different -- reading zero does not fail, it silently computes something
        less accurate than the term claims to (a second-order upwind reconstruction falling back to
        first order). Override to ``True`` for such a term, so an assembler with no reconstruction
        refuses it.

        Returns
        -------
        bool
            ``True`` when the answer on an orthogonal grid depends on the gradient.
        """
        return False


def declared_properties(terms: Iterable[DeclaredInputs]) -> set[str]:
    """The union of the properties a collection of terms reads.

    Parameters
    ----------
    terms : iterable of DeclaredInputs
        The terms an assembler holds.

    Returns
    -------
    set of str
        Every name any of them returns from :meth:`DeclaredInputs.requires`.
    """
    return {name for term in terms for name in term.requires()}
