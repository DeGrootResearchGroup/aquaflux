"""Run the case with PETSc made unreachable, so a completed march PROVES it was never used.

Reading imports cannot establish this: ``petsc4py`` is imported lazily, inside the one function that
builds a PETSc V-cycle, so whether a given configuration reaches it depends on which block inverses
were injected rather than on anything visible at module scope. Blocking it and marching turns the
question into an observation — and when the answer is "it still needs it", the traceback names the
call site, which a grep does not.

Two independent blocks, because either alone could be circumvented by the other path: the import
system refuses ``petsc4py`` outright, and the package's own accessor raises before it can ask.

Usage
-----
``BFS3D_FLOW_INVERSE=hostilu validation/run_case.sh validation/bfs3d_openfoam/petsc_free_march.py``

Every ``BFS3D_*`` setting the case reads applies here unchanged; this only installs the blocks and
then hands over to the case's own entry point.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))


class _BlockPetscImports:
    """Refuse ``petsc4py`` at the import system, leaving every other module alone."""

    def find_spec(self, name, path=None, target=None):
        if name == "petsc4py" or name.startswith("petsc4py."):
            raise ImportError(
                f"petsc4py is BLOCKED by petsc_free_march (attempted import of {name!r}). "
                "Reaching here means this configuration still needs PETSc; the traceback above "
                "names the site that asked."
            )
        return None


def _install() -> None:
    sys.meta_path.insert(0, _BlockPetscImports())

    # The package's own lazy accessor, blocked separately: it is the single door to PETSc inside
    # aquaflux, so failing here gives a far shorter traceback than an import error deep in a library.
    # ⚠️ The name is `_petsc`, and it is checked rather than assumed: an earlier version of this
    # harness patched `_require_petsc`, which reads like the accessor, does not exist, and would have
    # made every run look PETSc-free whether or not it was. A block that cannot fire is worse than no
    # block, so the attribute is asserted present before it is replaced.
    from aquaflux.solve import amg_preconditioner

    assert hasattr(amg_preconditioner, "_petsc"), (
        "aquaflux.solve.amg_preconditioner has no `_petsc` accessor -- it has been renamed, and this "
        "harness would silently stop blocking anything. Point it at the new name."
    )

    def _refuse():
        raise ImportError(
            "aquaflux asked for PETSc, but this run blocks it. The configuration is NOT PETSc-free: "
            "some block inverse fell back to build_amg_vcycle instead of an injected native one."
        )

    amg_preconditioner._petsc = _refuse
    print("[petsc-free] blocked: the petsc4py import, and amg_preconditioner._petsc", flush=True)


if __name__ == "__main__":
    _install()
    runpy.run_path(str(HERE / "compare.py"), run_name="__main__")
