"""The one place a pair budget becomes a receiver count."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import work
from aquaflux.radiation.work import in_passes, receivers_per_pass


def test_a_pass_takes_as_many_receivers_as_the_pairs_allow():
    assert receivers_per_pass(4_000_000, 7_516) == 532
    assert receivers_per_pass(4_000_000, 270_336) == 14


def test_a_receiver_bringing_more_pairs_than_the_limit_still_gets_a_pass():
    """Otherwise the chunk is zero receivers and the loop over them never ends."""
    assert receivers_per_pass(10, 1_000) == 1


def test_no_facets_is_one_pair_per_receiver_rather_than_a_division_by_zero():
    assert receivers_per_pass(10, 0) == 10


@pytest.mark.parametrize("limit", [0, -5])
def test_a_limit_of_no_pairs_is_refused(limit):
    with pytest.raises(ValueError, match="pair_limit must be at least 1"):
        receivers_per_pass(limit, 3)


def _watch_scans(monkeypatch) -> list[int]:
    """Record the length of every scan :func:`in_passes` runs."""
    lengths = []
    real = work.lax.scan

    def watched(step, init, xs, *args, **kwargs):
        lengths.append(len(xs))
        return real(step, init, xs, *args, **kwargs)

    monkeypatch.setattr(work.lax, "scan", watched)
    return lengths


@pytest.mark.parametrize(("n_points", "scans"), [(12, [3]), (14, [3, 1]), (2, [1])])
def test_every_chunk_runs_inside_a_scan_including_a_short_last_one(monkeypatch, n_points, scans):
    """A scan is compiled as a whole even when nothing around it is; a bare call of the body runs
    it one operation at a time, which is several times slower for the same pairs. So the short
    last chunk is a scan of one step, and the answer is every row in order."""
    lengths = _watch_scans(monkeypatch)
    points = jnp.arange(3.0 * n_points).reshape(n_points, 3)
    # Four points a pass: 8 pairs a pass at 2 pairs a point.
    out = in_passes(((points, 0),), 8, 2, lambda chunk: 2.0 * chunk[:, 0])
    assert lengths == scans
    np.testing.assert_array_equal(out, 2.0 * np.asarray(points)[:, 0])
