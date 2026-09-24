"""How much one pass over receivers may form, stated in the unit that actually costs memory.

Every loop in this package that visits receivers — the gather, the shadow-mask build, the ray test,
the winding-number check — forms arrays of one entry per **receiver-by-facet pair**, so a pass's
memory is its receivers times the emitter's facets. Bounding a pass by a number of *receivers*
therefore leaves its size to a number the caller did not choose: the same receiver count that is
a few hundred megabytes against a coarse lamp is gigabytes against a finely refined one, and
refining the emitter is exactly what a user does to make an answer more accurate. The failure
then is not an error but an out-of-memory kill, with no traceback naming the setting.

So a pass is bounded by pairs, and the receiver count follows from it and from the facet count
here, in one place.
"""

from __future__ import annotations

__all__ = ["DEFAULT_PAIR_LIMIT", "receivers_per_pass"]

#: Receiver-by-facet pairs one pass may form, when a caller does not say.
DEFAULT_PAIR_LIMIT = 4_000_000


def receivers_per_pass(pair_limit: int, per_receiver: int) -> int:
    """How many receivers a pass takes so that it forms at most ``pair_limit`` pairs.

    Parameters
    ----------
    pair_limit : int
        The pairs one pass may form. Must be at least one.
    per_receiver : int
        The pairs each receiver brings — the emitter's facet count, usually.

    Returns
    -------
    int
        At least one, so a single receiver against more facets than the limit still gets a pass
        of its own rather than none.

    Raises
    ------
    ValueError
        If ``pair_limit`` is less than one.
    """
    if pair_limit < 1:
        msg = f"pair_limit must be at least 1 receiver-by-facet pair; got {pair_limit}"
        raise ValueError(msg)
    return max(1, pair_limit // max(1, per_receiver))
