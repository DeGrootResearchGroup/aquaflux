"""Padded-row products as one fused, threaded loop, compiled by Numba and called from inside XLA programs.

A padded-row (ELLPACK) product ``y_i = sum_k values[i, k] x[columns[i, k]]`` written in JAX compiles on
the CPU to two kernels: a gather fusion that writes every product ``values * x[columns]`` out to memory
as one ``(n_rows * width)`` array, then a row-sum kernel that reads it back, which the CPU backend runs on
one thread. For a wide operator (a three-dimensional reach-3 stencil stores a few hundred entries a row)
that intermediate is hundreds of megabytes per product. A loop that keeps each row's sum in a register
reads every stored entry once and writes only the result, and threads over rows; on a 4-core CPU it
runs the same products several times faster.

This module holds that loop for the two forms of the index half
(:class:`~aquaflux.precondition.padded_rows.PaddedColumns`): columns held as an array, and columns
decoded from a cell pattern over full field-pair blocks, which the loop decodes itself, so the decoded
column array is never formed. Each loop is compiled by Numba and registered with XLA as a foreign
function (XLA's typed "FFI" custom-call interface), so it runs inside the compiled program exactly where
the JAX product would, with no crossing back to Python.

**When it is used** (:func:`fused_products`, decided once per process; :func:`padded_product_path`
reports the decision and its reason): on the CPU, when Numba's
threading layer is one that may be entered from several threads at once (OpenMP or TBB), since XLA can
run two products concurrently; the ``workqueue`` layer aborts the process on concurrent entry, and a
one-thread loop is no faster than XLA's form on narrow operators. ``AQUAFLUX_DISABLE_FUSED_PRODUCT=1``,
read at the first decision, turns it off, so the two forms can be compared in one checkout. On any other
backend, or where it is not used, the product is the JAX expression it replaces.

**What differs.** Each row is summed serially in slot order, which is not the order XLA's row sum uses,
so a result differs from the JAX form by rounding (a few units in the last place of the operator's
precision). The order is fixed, so the product is deterministic whatever the thread count. Slots past a
row's stored entries are skipped rather than multiplied by zero.

**How it talks to XLA.** A typed foreign function receives one ``XLA_FFI_CallFrame`` pointer and returns
an error pointer (null for success). The handlers below read the frame by the C layout of the FFI
version they declare, ``(0, 1)``, the oldest XLA still accepts: XLA asks a handler which version it was
written against when the handler is registered, and keeps the layout of every version it accepts. A
version XLA no longer accepts is refused at registration, and the product then falls back to the JAX
form with a warning. Registration also runs both products once against a numpy reference before
choosing them.
"""

from __future__ import annotations

import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numba
import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from aquaflux.warn_once import warn_once

#: The two foreign-function targets, one per form of the columns.
HELD_TARGET = "aquaflux_padded_product_held"
CELL_TARGET = "aquaflux_padded_product_cells"

#: The FFI version the handlers read the call frame by: (major, minor).
_FFI_VERSION = (0, 1)
#: Numba threading layers that may be entered from several threads at once.
_THREAD_SAFE_LAYERS = ("omp", "tbb")

# The C layout read below, in 8-byte words of a 64-bit build. ``XLA_FFI_CallFrame``: struct size,
# extension, api, context, stage (an enum in the low half of word 4), then ``XLA_FFI_Args`` (struct size,
# extension, count, types, the argument pointers) and ``XLA_FFI_Rets`` laid out the same way.
_FRAME_EXTENSION = 1
_FRAME_STAGE_HALF = 8
_FRAME_ARGUMENTS = 9
_FRAME_RESULTS = 14
_FRAME_WORDS = 15
# ``XLA_FFI_Buffer``: struct size, extension, dtype (low half of word 2), data, rank, dims.
_BUFFER_DTYPE_HALF = 4
_BUFFER_DATA = 3
_BUFFER_DIMS = 5
_BUFFER_WORDS = 6
# ``XLA_FFI_Metadata_Extension``: an internal extension (struct size, type in the low half of word 1,
# next), then the ``XLA_FFI_Metadata`` pointer. ``XLA_FFI_Metadata``: struct size, then the
# ``XLA_FFI_Api_Version`` (struct size, extension, major and minor as two 32-bit halves), then the traits.
_EXTENSION_TYPE_HALF = 2
_METADATA_EXTENSION_TYPE = 1
_EXTENSION_METADATA = 3
_API_VERSION_STRUCT_SIZE = 28
_STAGE_EXECUTE = 3
_F32 = 11
_F64 = 12


@intrinsic
def _pointer(typingctx, address):
    """The address ``address`` (an int64) as a ``void*``."""

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return types.voidptr(types.int64), codegen


@intrinsic
def _address(typingctx, pointer):
    """The ``void*`` ``pointer`` as an int64 address."""

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], context.get_value_type(types.int64))

    return types.int64(types.voidptr), codegen


@numba.njit(parallel=True, cache=True)
def _held_rows(columns, lengths, values, x, out):
    """``out = A x`` for columns held as an array: each row's stored slots, summed in slot order."""
    for row in numba.prange(values.shape[0]):
        total = values.dtype.type(0.0)
        for slot in range(lengths[row]):
            total += values[row, slot] * x[columns[row, slot]]
        out[row] = total


@numba.njit(parallel=True, cache=True)
def _cell_rows(cells, lengths, values, x, other_fields, n_other, out):
    """``out = A x`` for columns decoded from a cell pattern over full field-pair blocks.

    Row ``(f, d)`` (row field ``f``, cell ``d``) stores ``other_fields * lengths[d]`` entries: for each
    column field ``g`` in turn, one per entry of cell ``d``'s row of the cell pattern, at column
    ``g n_other + cells[d, k]``. The slots are those, in that order.
    """
    n_cells = cells.shape[0]
    for row in numba.prange(values.shape[0]):
        cell = row - (row // n_cells) * n_cells
        length = lengths[cell]
        total = values.dtype.type(0.0)
        slot = 0
        for field in range(other_fields):
            base = field * n_other
            for k in range(length):
                total += values[row, slot] * x[base + cells[cell, k]]
                slot += 1
        out[row] = total


@numba.njit(cache=True)
def _words(address, count):
    return numba.carray(_pointer(address), (count,), np.int64)


@numba.njit(cache=True)
def _halves(address, count):
    return numba.carray(_pointer(address), (count,), np.int32)


@numba.njit(cache=True)
def _answered_metadata(frame):
    """Answer XLA's registration-time question, which FFI version this handler reads; ``True`` if asked."""
    extension = _words(frame, _FRAME_WORDS)[_FRAME_EXTENSION]
    if extension == 0 or _halves(extension, 4)[_EXTENSION_TYPE_HALF] != _METADATA_EXTENSION_TYPE:
        return False
    metadata = _words(extension, 4)[_EXTENSION_METADATA]
    words = _words(metadata, 5)
    words[1] = _API_VERSION_STRUCT_SIZE
    words[2] = 0
    halves = _halves(metadata, 10)
    halves[6] = _FFI_VERSION[0]
    halves[7] = _FFI_VERSION[1]
    halves[8] = 0  # no traits
    return True


@numba.njit(cache=True)
def _buffer(frame, which, index):
    """``(dtype code, data address, dims address)`` of argument (``which`` 0) or result (1) ``index``."""
    words = _words(frame, _FRAME_WORDS)
    pointers = _words(words[_FRAME_RESULTS if which else _FRAME_ARGUMENTS], index + 1)
    buffer = pointers[index]
    fields = _words(buffer, _BUFFER_WORDS)
    return (
        _halves(buffer, 2 * _BUFFER_WORDS)[_BUFFER_DTYPE_HALF],
        fields[_BUFFER_DATA],
        fields[_BUFFER_DIMS],
    )


@numba.njit(cache=True)
def _executing(frame):
    return _halves(frame, 2 * _FRAME_WORDS)[_FRAME_STAGE_HALF] == _STAGE_EXECUTE


@numba.njit(cache=True)
def _held_call(frame):
    """The held-columns target, given its call frame's address.

    Arguments: columns ``int32[n, w]``, lengths ``int32[n]``, values ``[n, w]``, ``x``; result ``[n]``.
    The two precisions are two branches because an array's element type is fixed when it is compiled.
    """
    if _answered_metadata(frame) or not _executing(frame):
        return
    _, columns_at, shape_at = _buffer(frame, 0, 0)
    _, lengths_at, _ = _buffer(frame, 0, 1)
    dtype, values_at, _ = _buffer(frame, 0, 2)
    _, x_at, x_shape_at = _buffer(frame, 0, 3)
    _, out_at, _ = _buffer(frame, 1, 0)
    shape = _words(shape_at, 2)
    n_x = _words(x_shape_at, 1)[0]
    columns = numba.carray(_pointer(columns_at), (shape[0], shape[1]), np.int32)
    lengths = numba.carray(_pointer(lengths_at), (shape[0],), np.int32)
    if dtype == _F32:
        _held_rows(
            columns,
            lengths,
            numba.carray(_pointer(values_at), (shape[0], shape[1]), np.float32),
            numba.carray(_pointer(x_at), (n_x,), np.float32),
            numba.carray(_pointer(out_at), (shape[0],), np.float32),
        )
    else:
        _held_rows(
            columns,
            lengths,
            numba.carray(_pointer(values_at), (shape[0], shape[1]), np.float64),
            numba.carray(_pointer(x_at), (n_x,), np.float64),
            numba.carray(_pointer(out_at), (shape[0],), np.float64),
        )


@numba.njit(cache=True)
def _cell_call(frame):
    """The cell-pattern target, given its call frame's address.

    Arguments: cells ``int32[c, cw]``, lengths ``int32[c]``, values ``[n, w]``, ``x``, and ``int64[2]``
    holding the column fields and column cells; result ``[n]``.
    """
    if _answered_metadata(frame) or not _executing(frame):
        return
    _, cells_at, cells_shape_at = _buffer(frame, 0, 0)
    _, lengths_at, _ = _buffer(frame, 0, 1)
    dtype, values_at, shape_at = _buffer(frame, 0, 2)
    _, x_at, x_shape_at = _buffer(frame, 0, 3)
    _, counts_at, _ = _buffer(frame, 0, 4)
    _, out_at, _ = _buffer(frame, 1, 0)
    cells_shape = _words(cells_shape_at, 2)
    shape = _words(shape_at, 2)
    n_x = _words(x_shape_at, 1)[0]
    counts = _words(counts_at, 2)
    cells = numba.carray(_pointer(cells_at), (cells_shape[0], cells_shape[1]), np.int32)
    lengths = numba.carray(_pointer(lengths_at), (cells_shape[0],), np.int32)
    if dtype == _F32:
        _cell_rows(
            cells,
            lengths,
            numba.carray(_pointer(values_at), (shape[0], shape[1]), np.float32),
            numba.carray(_pointer(x_at), (n_x,), np.float32),
            counts[0],
            counts[1],
            numba.carray(_pointer(out_at), (shape[0],), np.float32),
        )
    else:
        _cell_rows(
            cells,
            lengths,
            numba.carray(_pointer(values_at), (shape[0], shape[1]), np.float64),
            numba.carray(_pointer(x_at), (n_x,), np.float64),
            counts[0],
            counts[1],
            numba.carray(_pointer(out_at), (shape[0],), np.float64),
        )


@functools.cache
def _handlers() -> tuple:
    """The two handlers ``(held, cells)``, compiled on first use rather than at import.

    Each takes the ``XLA_FFI_CallFrame`` pointer and returns a null error pointer, having answered a
    registration-time metadata question or run its loop. Cached, so the compiled functions XLA holds
    pointers to live as long as the process.
    """
    signature = types.voidptr(types.voidptr)

    @numba.cfunc(signature)
    def held(frame):
        _held_call(_address(frame))
        return _pointer(0)

    @numba.cfunc(signature)
    def cells(frame):
        _cell_call(_address(frame))
        return _pointer(0)

    return held, cells


def _capsule(address: int):
    """``address`` wrapped as the ``PyCapsule`` XLA's registration takes."""
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    return new(address, b"xla._CUSTOM_CALL_TARGET", None)


def held_product(columns, lengths, values, x):
    """``A x`` by the fused loop, for columns held as an array. CPU only; see :func:`fused_products`.

    Parameters
    ----------
    columns : jnp.ndarray
        Each slot's column, shape ``(n_rows, width)``, int32.
    lengths : jnp.ndarray
        Each row's stored entries, shape ``(n_rows,)``, int32.
    values : jnp.ndarray
        The slot values, shape ``(n_rows, width)``, float32 or float64.
    x : jnp.ndarray
        The vector, shape ``(n_cols,)``, in ``values``'s type.

    Returns
    -------
    jnp.ndarray
        ``A x``, shape ``(n_rows,)``, in ``values``'s type.

    Raises
    ------
    RuntimeError
        If XLA refused the targets (:func:`registered`).
    """
    _require_registered()
    return _held_call_of(columns, lengths, values, x)


def cell_product(cells, lengths, values, x, other_fields: int, n_other: int):
    """``A x`` by the fused loop, for columns decoded from a cell pattern. CPU only.

    Parameters
    ----------
    cells : jnp.ndarray
        Each cell row's columns, shape ``(n_cells, cell_width)``, int32.
    lengths : jnp.ndarray
        Each cell row's stored entries, shape ``(n_cells,)``, int32.
    values : jnp.ndarray
        The slot values, shape ``(fields * n_cells, width)``, float32 or float64.
    x : jnp.ndarray
        The vector, shape ``(other_fields * n_other,)``, in ``values``'s type.
    other_fields, n_other : int
        Column fields, and column cells.

    Returns
    -------
    jnp.ndarray
        ``A x``, shape ``(fields * n_cells,)``, in ``values``'s type.

    Raises
    ------
    RuntimeError
        If XLA refused the targets (:func:`registered`).
    """
    _require_registered()
    return _cell_call_of(cells, lengths, values, x, other_fields, n_other)


def _held_call_of(columns, lengths, values, x):
    """The foreign call :func:`held_product` makes, without the registration check."""
    return jax.ffi.ffi_call(HELD_TARGET, jax.ShapeDtypeStruct(values.shape[:1], values.dtype))(
        columns, lengths, values, x
    )


def _cell_call_of(cells, lengths, values, x, other_fields, n_other):
    """The foreign call :func:`cell_product` makes, without the registration check."""
    counts = jnp.asarray([other_fields, n_other], dtype=jnp.int64)
    return jax.ffi.ffi_call(CELL_TARGET, jax.ShapeDtypeStruct(values.shape[:1], values.dtype))(
        cells, lengths, values, x, counts
    )


def _require_registered() -> None:
    if not registered():
        raise RuntimeError(
            "the fused padded-row products are not registered with XLA in this process (see the "
            "warning issued when registration was refused)."
        )


def routed(fused, plain, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """``fused(values, x)`` on the CPU where :func:`fused_products` allows it, else ``plain(values, x)``.

    The fused loop takes one floating type for both operands; a mixed pair (which promotes in the JAX
    form) and an integer pair take ``plain``. On a backend other than the CPU, ``plain`` is what is
    compiled.
    """
    same_float = values.dtype == x.dtype and values.dtype in (jnp.float32, jnp.float64)
    if not (same_float and fused_products()):
        return plain(values, x)
    # Wrapped, since a bound method of a module holding arrays cannot be hashed as a branch.
    return jax.lax.platform_dependent(
        values, x, cpu=lambda v, y: fused(v, y), default=lambda v, y: plain(v, y)
    )


def fused_products() -> bool:
    """Whether padded-row products run as the fused loop in this process. Decided once.

    ``True`` when the CPU backend is present, ``AQUAFLUX_DISABLE_FUSED_PRODUCT`` is unset, Numba's
    threading layer may be entered from several threads at once, and both targets registered and agreed
    with a numpy product on a small operator.
    """
    return _decision()[0]


def padded_product_path() -> dict[str, str] | None:
    """Which form this process's padded-row products took, and why, for a run's record.

    Returns
    -------
    dict or None
        ``{"form": "fused" | "gathered", "reason": ...}``, the reason naming the threading layer the loop
        runs on or what kept the gathered form (the environment switch, a threading layer that two threads
        may not enter at once, a refused registration, a default backend other than the CPU). ``None``
        when no padded-row product has been formed in this process, so the question never came up.
    """
    if _decision.cache_info().currsize == 0:
        return None
    fused, reason = _decision()
    backend = jax.default_backend()
    if fused and backend != "cpu":
        fused, reason = (
            False,
            f"the default backend is {backend}, where the gathered form is compiled",
        )
    return {"form": "fused" if fused else "gathered", "reason": reason}


@functools.cache
def _decision() -> tuple[bool, str]:
    """:func:`_decide`, once per process."""
    return _decide()


def _decide() -> tuple[bool, str]:
    """Whether the fused loop may be used here, and the reason, in words a run's record can carry."""
    if os.environ.get("AQUAFLUX_DISABLE_FUSED_PRODUCT"):
        return False, "AQUAFLUX_DISABLE_FUSED_PRODUCT is set"
    try:
        jax.devices("cpu")
    except RuntimeError:
        return False, "there is no CPU backend"
    # The threading layer is settled by the first parallel launch; this one also compiles the loop.
    columns = np.zeros((2, 1), dtype=np.int32)
    _held_rows(columns, np.ones(2, np.int32), np.ones((2, 1)), np.ones(1), np.empty(2))
    layer = numba.threading_layer()
    if layer not in _THREAD_SAFE_LAYERS:
        return False, (
            f"Numba's threading layer is {layer}, which two threads may not enter at once; "
            "the loop needs OpenMP or TBB"
        )
    if not registered():
        return False, "XLA refused the fused targets or they disagreed with numpy (see the warning)"
    return True, f"Numba's threading layer is {layer}, {numba.get_num_threads()} threads"


@functools.cache
def registered() -> bool:
    """Register both targets with XLA's CPU backend once, and check them; ``False`` (with a warning) if
    either is refused or disagrees with a numpy product. Any thread count and threading layer.
    """
    # Registering before the backend exists defers the check to the backend's start, where a refusal
    # would stop the backend itself; started first, a refusal raises here.
    jax.devices("cpu")
    try:
        held, cells = _handlers()
        jax.ffi.register_ffi_target(HELD_TARGET, _capsule(held.address), platform="cpu")
        jax.ffi.register_ffi_target(CELL_TARGET, _capsule(cells.address), platform="cpu")
    except jax.errors.JaxRuntimeError as error:
        warn_once(
            "fused_product.refused",
            f"XLA refused the fused padded-row products ({error}); using the JAX form.",
        )
        return False
    disagreement = _self_check()
    if disagreement is not None:
        warn_once(
            "fused_product.disagreed",
            f"the fused padded-row products disagree with numpy ({disagreement}); using the JAX form.",
        )
        return False
    return True


def _self_check() -> str | None:
    """Both products against numpy on a small operator of each form, both precisions; ``None`` if equal."""
    rng = np.random.default_rng(0)
    cells = np.array([[1, 0, 2], [0, 2, 0], [2, 1, 0]], dtype=np.int32)
    cell_lengths = np.array([3, 2, 1], dtype=np.int32)
    other_fields, n_other, fields = 2, 3, 2
    width = other_fields * cells.shape[1]
    n_rows = fields * cells.shape[0]
    columns = np.zeros((n_rows, width), dtype=np.int32)
    lengths = np.zeros(n_rows, dtype=np.int32)
    for row in range(n_rows):
        cell = row % cells.shape[0]
        run = [
            g * n_other + c for g in range(other_fields) for c in cells[cell, : cell_lengths[cell]]
        ]
        columns[row, : len(run)] = run
        lengths[row] = len(run)
    with jax.ensure_compile_time_eval():
        for dtype in (np.float32, np.float64):
            values = rng.standard_normal((n_rows, width)).astype(dtype)
            values[np.arange(width)[None, :] >= lengths[:, None]] = 0
            x = rng.standard_normal(other_fields * n_other).astype(dtype)
            expected = np.sum(values.astype(np.float64) * x[columns], axis=1)
            held = np.asarray(jax.jit(_held_call_of)(columns, lengths, values, x))
            cell = np.asarray(
                jax.jit(_cell_call_of, static_argnums=(4, 5))(
                    cells, cell_lengths, values, x, other_fields, n_other
                )
            )
            magnitude = np.max(
                np.sum(np.abs(values.astype(np.float64)) * np.abs(x[columns]), axis=1)
            )
            tolerance = 8 * np.finfo(dtype).eps * magnitude
            for name, got in (("held", held), ("cells", cell)):
                if got.dtype != dtype or not np.allclose(got, expected, rtol=0, atol=tolerance):
                    return f"{name} columns, {np.dtype(dtype).name}"
    return None
