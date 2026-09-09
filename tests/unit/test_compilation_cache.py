"""The persistent compilation cache is configured -- and bounded -- at ``import aquaflux``.

JAX's on-disk compilation cache keeps expensive compilations across runs, but its default
``jax_compilation_cache_max_size`` is ``-1``, which its least-recently-used (LRU) implementation
reads as "no eviction": such a cache never prunes anything and only ever grows, reaching tens of
gigabytes of month-old entries on a long-lived checkout. ``aquaflux/__init__.py`` therefore sets a
byte bound, which is what turns eviction on.

The bound has one sharp edge, and it is why these tests exist rather than a single assertion that a
number was set: JAX takes an inter-process lock through ``filelock`` when evicting, and with a bound
set but that package missing, every cache read and write fails with a warning and the cache stores
*nothing*. A silently dead cache is worse than an unbounded one, so the package only sets the bound
where eviction is actually supported, and the subprocess tests below pin that contract from both
sides -- they assert the pairing, not one branch, so they hold in an environment with ``filelock``
and in one without it.

Import-time configuration is process-global, so the fresh-import behaviour is checked in a
subprocess: an in-process assertion would depend on whatever else had already configured JAX.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from aquaflux import _cache_eviction_is_supported, _cache_max_bytes

_GIB = 1024**3


def _fresh_import(code: str, env_extra: dict[str, str]) -> str:
    """Run ``code`` in a clean interpreter that imports aquaflux, and return its stdout.

    Parameters
    ----------
    code : str
        Python source to run after the environment is set.
    env_extra : dict of str to str
        Environment entries to add for the child.

    Returns
    -------
    str
        The child's stripped stdout.
    """
    import os

    env = {**os.environ, **env_extra}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 10 * _GIB),  # unset -> the default bound
        ("", 10 * _GIB),  # empty -> the default bound
        ("   ", 10 * _GIB),  # whitespace -> the default bound
        ("2", 2 * _GIB),
        ("0.5", _GIB // 2),
        ("-1", -1),  # negative -> unbounded, mirroring JAX's own convention
        ("-0.5", -1),
        ("0", 0),  # taken literally: zero must not be read as "no bound"
    ],
)
def test_the_size_bound_is_read_from_gibibytes(raw: str | None, expected: int) -> None:
    """The environment override is read in gibibytes, with a negative value meaning unbounded."""
    assert _cache_max_bytes(raw) == expected


def test_an_unparseable_bound_falls_back_to_the_default() -> None:
    """A typo in the override must not decide how much disk the cache may use."""
    assert _cache_max_bytes("ten gigabytes") == 10 * _GIB
    assert _cache_max_bytes("1e") == 10 * _GIB


def test_the_default_is_a_real_bound_and_not_jax_s_unbounded_sentinel() -> None:
    """The regression this guards: an unset override must not leave the cache unpruned.

    ``-1`` is JAX's "no eviction" sentinel, and it is also the value the cache has when nothing
    sets it -- so a default that computed to ``-1`` would silently restore the unbounded growth
    the bound exists to stop, while still looking configured.
    """
    assert _cache_max_bytes(None) > 0


def test_eviction_support_follows_whether_filelock_can_be_found(monkeypatch) -> None:
    """Support is decided by locating ``filelock``, since that is what JAX needs to evict."""
    import aquaflux

    monkeypatch.setattr(aquaflux._importlib_util, "find_spec", lambda name: None)
    assert not _cache_eviction_is_supported()

    monkeypatch.setattr(aquaflux._importlib_util, "find_spec", lambda name: object())
    assert _cache_eviction_is_supported()


def test_a_fresh_import_bounds_the_cache_exactly_when_it_can_evict(tmp_path) -> None:
    """The bound and the ability to enforce it are set together, never one without the other.

    Both halves matter. A bound without ``filelock`` disables the cache entirely, and no bound
    with it lets the cache grow forever, so this asserts the pairing rather than either value.
    """
    code = (
        "import aquaflux, jax;"
        "supported = aquaflux._cache_eviction_is_supported();"
        "size = jax.config.jax_compilation_cache_max_size;"
        "print(supported, size > 0, size)"
    )
    supported, bounded, _ = _fresh_import(
        code, {"AQUAFLUX_COMPILATION_CACHE_DIR": str(tmp_path / "jax")}
    ).split()
    assert supported == bounded


def test_a_fresh_import_honours_the_bound_override(tmp_path) -> None:
    """An explicit override reaches JAX wherever the bound can be enforced at all.

    Written as a pairing rather than a skip: where eviction is unsupported the override is
    deliberately ignored, and that is the behaviour worth pinning there -- a skip would leave
    the no-``filelock`` environment with no coverage of the override at all.
    """
    code = "import aquaflux, jax; print(jax.config.jax_compilation_cache_max_size)"
    out = int(
        _fresh_import(
            code,
            {
                "AQUAFLUX_COMPILATION_CACHE_DIR": str(tmp_path / "jax"),
                "AQUAFLUX_COMPILATION_CACHE_MAX_GIB": "3",
            },
        )
    )
    assert out == (3 * _GIB if _cache_eviction_is_supported() else -1)


def test_the_cache_can_be_disabled_entirely(tmp_path) -> None:
    """The opt-out leaves the cache directory unset, for a read-only or ephemeral filesystem."""
    code = "import aquaflux, jax; print(repr(jax.config.jax_compilation_cache_dir))"
    out = _fresh_import(
        code,
        {
            "AQUAFLUX_DISABLE_COMPILATION_CACHE": "1",
            "AQUAFLUX_COMPILATION_CACHE_DIR": str(tmp_path / "jax"),
        },
    )
    assert str(tmp_path) not in out
