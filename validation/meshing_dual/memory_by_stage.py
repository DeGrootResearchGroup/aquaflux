"""Memory and time of each stage of ``CadMesher.mesh`` on the Sozzi water, by mesh size.

The Sozzi water is meshed as ``mapped_layers_sozzi.py`` meshes it with its walls mapped (the lamp's
and the pipes' cylinders structured, five layers on every wall, the pipes' cores O-grids), and the
mesh's geometry is computed after, as a solve would, with its closure and smallest cell printed. Each
stage of the mesher is wrapped, and for each the harness prints its time and this process's
physical footprint (the memory the operating system charges it, compressed pages included -- not
its resident set) at the stage's start, its peak during the stage, and at its end. Netgen runs in a
child process; its peak resident set is printed beside the stage that ran it, since the parent's
footprint is held while it runs. The peak during a stage is sampled every 5 ms and also read from
the kernel's lifetime maximum, so a peak the sampler misses (a thread holding the interpreter
lock) is still seen when it is a new maximum.

Each size runs in its own process, one after another, so each starts from nothing. Run with
``MallocLargeCache=0``: by default macOS's allocator keeps freed large blocks charged to the process,
so a footprint reads what was once allocated as well as what is live. macOS only (it
reads the footprint through ``proc_pid_rusage``). Run in an environment with ``cadquery-ocp-novtk``
and with ``netgen-mesher`` in the Python ``MESHING_NETGEN_PYTHON`` names (the running one by default).
``MESHING_NO_PIPE_CORES=1`` leaves the pipes' cores to the tetrahedra, and ``MESHING_TRIANGLE_WALLS=1``
meshes every wall with triangles (prism layers, no hexahedra)::

    python validation/meshing_dual/memory_by_stage.py 0.02 0.01 0.007
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import resource
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

GIB = 2.0**30
_RUSAGE_INFO_V4 = 4
_PHYS_FOOTPRINT = 7  # the uint64 fields of rusage_info_v4 after its 16-byte uuid
_LIFETIME_MAX_PHYS_FOOTPRINT = 28


class _Footprint:
    """This process's physical footprint, now and its lifetime maximum, in bytes."""

    def __init__(self) -> None:
        self._libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self._buffer = (ctypes.c_uint64 * 40)()  # rusage_info_v4: a 16-byte uuid, then 35 fields

    def _read(self) -> ctypes.Array:
        if self._libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(self._buffer)):
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return self._buffer

    def now(self) -> int:
        return int(self._read()[2 + _PHYS_FOOTPRINT])

    def lifetime_max(self) -> int:
        return int(self._read()[2 + _LIFETIME_MAX_PHYS_FOOTPRINT])


class _Stages:
    """Wraps functions so each call is reported as a stage: its time, footprints and child peak."""

    def __init__(self) -> None:
        self.footprint = _Footprint()
        self._peak = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self) -> None:
        while True:
            now = self.footprint.now()
            with self._lock:
                self._peak = max(self._peak, now)
            time.sleep(0.005)

    def wrap(self, owner, name: str, label: str) -> None:
        original = getattr(owner, name)
        raw = owner.__dict__.get(name) if isinstance(owner, type) else None
        stages = self

        def wrapped(*args, **kwargs):
            return stages.run(label, lambda: original(*args, **kwargs))

        # A static or class method, read off the class, is already the callable to wrap.
        plain = not isinstance(raw, (staticmethod, classmethod))
        setattr(owner, name, wrapped if plain else staticmethod(wrapped))

    def run(self, label: str, call):
        start = self.footprint.now()
        lifetime = self.footprint.lifetime_max()
        children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        with self._lock:
            outer, self._peak = self._peak, start  # a stage within another keeps the outer's peak
        began = time.perf_counter()
        result = call()
        seconds = time.perf_counter() - began
        end = self.footprint.now()
        with self._lock:
            peak = max(self._peak, end)
            self._peak = max(outer, peak)
        after = self.footprint.lifetime_max()
        if after > lifetime:
            peak = max(peak, after)
        child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # bytes on macOS
        child_note = f"  Netgen child peak RSS {child / GIB:5.2f}" if child > children else ""
        print(
            f"  {label:<22} {seconds:7.1f} s  footprint start {start / GIB:5.2f}  "
            f"peak {peak / GIB:5.2f}  end {end / GIB:5.2f} GiB{child_note}",
            flush=True,
        )
        return result


def _one_size(size: float) -> None:
    import numpy as np
    from aquaflux.io.cad import Placement, read_step
    from aquaflux.mesh import Mesh, closed_cell_residual
    from aquaflux.meshing import (
        CadMesher,
        NetgenProcess,
        PolyhedralDual,
        PrismLayers,
        PyramidCaps,
        cad_mesher,
        dual,
    )
    from mapped_layers_sozzi import DRAWING, LAYERED, LAYERS, PATCHES, WATER

    stages = _Stages()
    stages.wrap(NetgenProcess, "triangulate", "Netgen surface")
    stages.wrap(CadMesher, "_mapped", "mapped walls")
    stages.wrap(PrismLayers, "extrude", "layers")
    stages.wrap(cad_mesher, "fill_conduits", "pipe and duct cores")
    stages.wrap(PyramidCaps, "on", "pyramid caps")
    stages.wrap(CadMesher, "_filled", "Netgen fill")
    stages.wrap(cad_mesher, "element_mesh", "primal Mesh")
    stages.wrap(cad_mesher._Columns, "of", "layer columns")
    stages.wrap(PolyhedralDual, "build_traced", "dual")
    stages.wrap(cad_mesher, "_onto_the_drawing", "onto the drawing")
    # Within the dual, and every mesh geometry and mesh assembly wherever it is called.
    stages.wrap(dual._PrimalTables, "of", ". dual: tables")
    stages.wrap(PolyhedralDual, "_features", ". dual: features")
    stages.wrap(dual._Walks, "of", ". dual: walks")
    stages.wrap(dual._Pieces, "of", ". dual: pieces")
    stages.wrap(dual, "_assemble", ". dual: assembly")
    stages.wrap(Mesh, "geometry", ". . Mesh.geometry")
    stages.wrap(Mesh, "from_csr", ". . Mesh.from_csr")

    model = read_step(DRAWING, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    netgen = NetgenProcess(python=os.environ.get("MESHING_NETGEN_PYTHON", sys.executable))
    mesher = CadMesher(max_size=size, feature_angle=0.785, layers=LAYERS, netgen=netgen)
    pipes = () if os.environ.get("MESHING_NO_PIPE_CORES") else None
    mapped = () if os.environ.get("MESHING_TRIANGLE_WALLS") else ("lamp", "pipes")
    notes = ["no pipe cores"] * (pipes == ()) + ["triangle walls"] * (mapped == ())
    print(f"maxh {size}" + "".join(f"  ({note})" for note in notes), flush=True)
    began = time.perf_counter()
    mesh = stages.run(
        "mesh (whole)",
        lambda: mesher.mesh(model, WATER, PATCHES, layered=LAYERED, mapped=mapped, pipes=pipes),
    )
    meshed = time.perf_counter() - began
    geometry = stages.run("dual geometry", lambda: mesh.geometry())
    np.asarray(geometry.cell.volume)
    incidences = int(np.asarray(mesh.face_nodes.offsets)[-1])
    print(
        f"  cells {mesh.n_cells}  faces {mesh.n_faces}  face-node incidences {incidences}  "
        f"meshed in {meshed:.1f} s  lifetime peak footprint "
        f"{stages.footprint.lifetime_max() / GIB:.2f} GiB",
        flush=True,
    )
    # After the footprints: every cell closed, none inside out.
    closure = float(np.max(np.asarray(closed_cell_residual(mesh)))) / float(
        np.max(np.asarray(geometry.face.area))
    )
    print(
        f"  closure {closure:.1e} (largest face area)  min cell volume "
        f"{float(np.min(np.asarray(geometry.cell.volume))):.2e}",
        flush=True,
    )


def main(sizes: list[float]) -> None:
    from peak_footprint import run_with_footprint

    for size in sizes:
        ran = run_with_footprint([sys.executable, "-u", __file__, "--one", str(size)])
        if ran.returncode:
            raise SystemExit(f"maxh {size} failed with status {ran.returncode}")
        print(
            f"  whole process peak footprint (time -l) {ran.peak_footprint_bytes / GIB:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        _one_size(float(sys.argv[2]))
    else:
        main([float(argument) for argument in sys.argv[1:]] or [0.02, 0.01, 0.007])
