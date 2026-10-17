"""Where a line meets a torus, decided in exact rational arithmetic.

The reference for :class:`aquaflux.solids.Torus`. Every float the body is given is a rational
number, so the quartic whose negative stretches are the line's inside of the ring,

    ((|x|² + R² - r²)² - 4 R² rho²)  with  x = origin - centre + t direction,

has exact rational coefficients, and its real roots can be counted and isolated exactly by a
Sturm sequence (Sturm, 1829) -- no rounding anywhere, so the answer is the true one *for the
inputs as stored*. That is the only reference that can judge a line grazing the tube: any floating
point method, however careful, is itself what is being judged there.

The axis is read as the direction it is stored as, not assumed to be of unit length, which is what
the body's own normalization approximates; tests that need the two to coincide exactly use a
coordinate axis.

Plain Python and ``fractions``, so it needs nothing installed and runs anywhere the suite does.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise


def _mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _add(a, b):
    out = [Fraction(0)] * max(len(a), len(b))
    for i, x in enumerate(a):
        out[i] += x
    for i, y in enumerate(b):
        out[i] += y
    return out


def _scale(a, s):
    return [x * s for x in a]


def _trim(a):
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return a


def _evaluate(p, x):
    value = Fraction(0)
    for coefficient in reversed(p):
        value = value * x + coefficient
    return value


def _derivative(p):
    return [i * c for i, c in enumerate(p)][1:] or [Fraction(0)]


def _remainder(a, b):
    a = list(a)
    while len(a) >= len(b):
        factor = a[-1] / b[-1]
        shift = len(a) - len(b)
        for i, c in enumerate(b):
            a[i + shift] -= factor * c
        a.pop()
        if not a:
            return [Fraction(0)]
    return _trim(a)


def quartic(origin, direction, centre, axis, ring_radius, tube_radius):
    """The quartic in ``t``, lowest power first, negative where the line is inside the ring."""
    origin, direction = (
        [Fraction(float(v)) for v in origin],
        [Fraction(float(v)) for v in direction],
    )
    centre, axis = [Fraction(float(v)) for v in centre], [Fraction(float(v)) for v in axis]
    ring, tube = Fraction(float(ring_radius)), Fraction(float(tube_radius))
    x = [[o - c, d] for o, c, d in zip(origin, centre, direction, strict=True)]
    squared = [Fraction(0)]
    along = [Fraction(0)]
    for component, k in zip(x, axis, strict=True):
        squared = _add(squared, _mul(component, component))
        along = _add(along, _scale(component, k))
    axis_squared = sum(k * k for k in axis)
    radial = _add(squared, _scale(_mul(along, along), -1 / axis_squared))
    shifted = _add(squared, [ring * ring - tube * tube])
    return _trim(_add(_mul(shifted, shifted), _scale(radial, -4 * ring * ring)))


def _sturm(p):
    chain = [p, _derivative(p)]
    while len(chain[-1]) > 1 or chain[-1][0] != 0:
        rest = _remainder(chain[-2], chain[-1])
        if len(rest) == 1 and rest[0] == 0:
            break
        chain.append(_scale(rest, -1))
    return chain


def _changes(signs):
    signs = [s for s in signs if s != 0]
    return sum(1 for a, b in pairwise(signs) if a != b)


def _count(chain, x):
    """Sign changes of the chain at ``x``; ``x`` may be ``+-inf`` given as the strings."""
    if x == "-inf":
        return _changes([(-1) ** (len(p) - 1) * (1 if p[-1] > 0 else -1) for p in chain])
    if x == "+inf":
        return _changes([1 if p[-1] > 0 else -1 for p in chain])
    return _changes([(v > 0) - (v < 0) for v in (_evaluate(p, x) for p in chain)])


def has_real_root(p):
    """Whether the polynomial has a real root -- for a line, whether it meets the ring at all."""
    chain = _sturm(p)
    return _count(chain, "-inf") - _count(chain, "+inf") > 0


def real_roots(p, tolerance=Fraction(1, 2**64)):
    """The distinct real roots, each to within ``tolerance`` times its own size plus one."""
    chain = _sturm(p)
    bound = 1 + max(abs(c / p[-1]) for c in p[:-1])
    pending, isolated = [(-bound, bound)], []
    while pending:
        low, high = pending.pop()
        count = _count(chain, low) - _count(chain, high)
        if count == 0:
            continue
        if count == 1:
            isolated.append((low, high))
            continue
        middle = (low + high) / 2
        pending += [(low, middle), (middle, high)]
    roots = []
    for low, high in isolated:
        while high - low > tolerance * (1 + abs(low)):
            middle = (low + high) / 2
            chain_low = _count(chain, low) - _count(chain, middle)
            low, high = (low, middle) if chain_low == 1 else (middle, high)
        roots.append((low + high) / 2)
    return sorted(roots)


def inside_intervals(origin, direction, centre, axis, ring_radius, tube_radius):
    """The stretches of ``t`` where the line is inside the whole ring, as float pairs, in order."""
    p = quartic(origin, direction, centre, axis, ring_radius, tube_radius)
    roots = real_roots(p)
    points = [roots[0] - 1, *roots, roots[-1] + 1] if roots else []
    intervals = []
    for left, right in pairwise(points):
        if _evaluate(p, (left + right) / 2) < 0:
            if intervals and intervals[-1][1] == left:  # touched from inside: one stretch
                left = intervals.pop()[0]
            intervals.append((left, right))
    return [(float(a), float(b)) for a, b in intervals]
