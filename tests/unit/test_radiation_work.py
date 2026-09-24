"""The one place a pair budget becomes a receiver count."""

from __future__ import annotations

import pytest
from aquaflux.radiation.work import receivers_per_pass


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
