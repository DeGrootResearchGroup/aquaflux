"""Values for the nodes outside the region, so the interpolant beside the boundary follows the field.

A node outside the region -- or inside it by less than the standoff, which is treated alike -- is
never sampled, yet its hat reaches into the region: every cell the boundary cuts has such a corner.
Left at what the coarser nodes make it, it says nothing about the field beside the boundary, and every
node finer than it would carry, in its surplus, a correction of that arbitrary value. So it is given a
**ghost** value, entered in the hierarchy like a sample, chosen from the samples around it -- and
chosen again, from finer samples, whenever finer cells appear around it. The value is a least-squares
compromise between two kinds of equation:

- **along each edge to a sampled neighbour** -- a neighbour along an axis, one cell away at the scale
  of the finest cell around the node -- the segment leaves the region somewhere, and a cut point lies
  a standoff from the boundary on it, with the field sampled there. The interpolant along that edge is
  a straight line between the neighbour's value and the ghost, and the ghost that makes the line pass
  through the cut point's value is one equation;
- **toward a linear extrapolation**, weighted lightly -- a linear fit to the samples around the node
  (nodes and cut points within two cells of that scale, the nearer weighing more), evaluated at the
  node. That decides a node with no such edge (one whose sampled neighbours are all diagonal to it, at
  a corner of the region), and steadies one whose cut point lies so near its neighbour that the slope
  between them is mostly the field's small-scale noise.

The ghost is then held within the range of the samples it was fitted to, widened by that range on
either side: it may carry their trend past them, but not amplify their noise without bound.
"""

from __future__ import annotations

import numpy as np

#: Weight of the pull toward the linear extrapolation, against weight ``t`` on the equation of an edge
#: whose cut point lies a fraction ``t`` of the way from the sampled neighbour to the node. An edge is
#: followed closely while ``t`` is well above this, and the extrapolation takes over below it -- where
#: the cut point lies so near the neighbour that the slope between them is mostly the field's noise.
EXTRAPOLATION_WEIGHT = 0.1


#: Distance, in cells of the node's level, inside which a sample weighs no more in the extrapolation
#: than one at that distance: a cut point a standoff from the node would otherwise decide it alone.
NEAREST = 0.5


def _batched(owner: np.ndarray, n: int, *columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scatter per-item rows into a batch, one row of the batch per owner, padded with zeros."""
    counts = np.bincount(owner, minlength=n)
    order = np.argsort(owner, kind="stable")
    local = np.empty(len(owner), dtype=np.int64)
    local[order] = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
    width = max(int(counts.max()) if len(owner) else 0, 1)
    out = []
    for column in columns:
        batch = np.zeros((n, width, *np.shape(column)[1:]))
        batch[owner, local] = column
        out.append(batch)
    return tuple(out)


def ghost_values(
    n: int,
    edge_owner: np.ndarray,
    edge_t: np.ndarray,
    edge_neighbour: np.ndarray,
    edge_cut: np.ndarray,
    data_owner: np.ndarray,
    data_offset: np.ndarray,
    data_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The ghost of each of ``n`` nodes outside the region.

    Parameters
    ----------
    n : int
        How many nodes.
    edge_owner : np.ndarray of int, shape ``(e,)``
        For each edge to a sampled neighbour that has a cut point, the node it belongs to.
    edge_t : np.ndarray, shape ``(e,)``
        Where the cut point lies, as a fraction of the way from the neighbour (``0``) to the node
        (``1``).
    edge_neighbour, edge_cut : np.ndarray, shape ``(e,)``
        The value at the neighbour and at the cut point.
    data_owner : np.ndarray of int, shape ``(d,)``
        For each sample the extrapolation is fitted to, the node it belongs to.
    data_offset : np.ndarray, shape ``(d, 3)``
        Where that sample lies relative to the node, in units of the node's own cell size.
    data_value : np.ndarray, shape ``(d,)``
        The value there.

    Returns
    -------
    ghosts : np.ndarray, shape ``(n,)``
    informed : np.ndarray of bool, shape ``(n,)``
        Which nodes had any sample to go on; the others' ghosts are zero and mean nothing.
    """
    informed = np.zeros(n, dtype=bool)
    informed[data_owner] = True
    informed[edge_owner] = True
    # The extrapolation: a weighted least-squares linear fit per node, about the samples' weighted
    # mean, so that where they do not determine the gradient (fewer than four, or all in a plane) the
    # gradient left undetermined is zero rather than the value shrunk: one sample extrapolates as a
    # constant. Weighted by inverse squared distance, so the nearest samples decide and the farther
    # ones only fill in a direction the near ones leave open.
    weight = 1.0 / np.maximum(np.sum(data_offset**2, axis=1), NEAREST**2)
    total = np.maximum(np.bincount(data_owner, weights=weight, minlength=n), np.finfo(float).tiny)
    mean_offset = (
        np.stack(
            [
                np.bincount(data_owner, weights=weight * data_offset[:, a], minlength=n)
                for a in range(3)
            ],
            axis=1,
        )
        / total[:, None]
    )
    mean_value = np.bincount(data_owner, weights=weight * data_value, minlength=n) / total
    root = np.sqrt(weight)
    centred = root[:, None] * (data_offset - mean_offset[data_owner])
    design_batch, value_batch = _batched(
        data_owner, n, centred, root * (data_value - mean_value[data_owner])
    )
    gradient = np.einsum("nij,nj->ni", np.linalg.pinv(design_batch, rtol=1e-10), value_batch)
    extrapolated = mean_value - np.sum(gradient * mean_offset, axis=1)
    # The edges: (1 - t) neighbour + t ghost = cut, in least squares with the pull toward it.
    pull = EXTRAPOLATION_WEIGHT**2
    numerator = pull * extrapolated + np.bincount(
        edge_owner, weights=edge_t * (edge_cut - (1.0 - edge_t) * edge_neighbour), minlength=n
    )
    denominator = pull + np.bincount(edge_owner, weights=edge_t**2, minlength=n)
    ghosts = numerator / denominator
    # A ghost may extend the trend of the samples it was fitted to by as much as they themselves span,
    # and no further: samples bunched close together carry their small-scale noise into the slope, and
    # extrapolated a cell beyond them it can grow without bound.
    low = np.full(n, np.inf)
    high = np.full(n, -np.inf)
    for owner, value in (
        (data_owner, data_value),
        (edge_owner, edge_cut),
        (edge_owner, edge_neighbour),
    ):
        np.minimum.at(low, owner, value)
        np.maximum.at(high, owner, value)
    span = high - low
    ghosts = np.clip(ghosts, low - span, high + span)
    return np.where(informed, ghosts, 0.0), informed
