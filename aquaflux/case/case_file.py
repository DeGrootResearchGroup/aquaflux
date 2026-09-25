"""Reading a case file, writing one, and checking one against its mesh.

A case file is YAML (a plain-text format of nested mappings and lists). Only the parse is
YAML-specific: what the document means is read from the parsed mapping by
:func:`~aquaflux.case.case_spec_from_mapping`, so every setting is checked where it appears.

The parse follows the YAML 1.2 rules for plain scalars rather than the 1.1 rules the common Python
parser defaults to, because the older rules misread ordinary case settings without a word:
``1e-5`` would be a string, ``no`` and ``on`` would be booleans, and ``010`` would be eight. A mapping
that names one key twice -- two ``inlet`` entries under ``boundaries``, say -- is refused rather than
keeping whichever came last.

Checking a case is the cheap half of loading it: the file is read and validated, the mesh's topology
is read and validated, and the case is checked against that topology. No geometry is computed and no
equation is built, so a file can be checked in about the time its mesh takes to read.
:meth:`CheckedCase.build` is the other half: the geometry, then the equations.

What a case builds is its problem -- the assembler an initializer and a solve take -- and never a
built solve. :meth:`CheckedCase.solve` solves that problem with the case's solver settings: a solve's
frozen preconditioner is fitted to a state, and a march re-fits it from states the file never sees (at
each station of a viscosity ramp, at each refresh), so what the file holds is the settings every such
fit is made with.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import yaml

from aquaflux.mesh import Mesh

from .solver import solver_for
from .spec import CaseSpec, case_spec_from_mapping, case_spec_to_mapping

__all__ = ["CaseFile", "CheckedCase", "read_case", "write_case"]

_BOOL_TAG = "tag:yaml.org,2002:bool"
_FLOAT_TAG = "tag:yaml.org,2002:float"
_INT_TAG = "tag:yaml.org,2002:int"


class _CaseLoader(yaml.SafeLoader):
    """A safe YAML loader reading plain scalars by the YAML 1.2 core rules, and refusing duplicate keys."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"the key {key!r} appears twice in one mapping", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


# Replace the 1.1 resolvers for booleans, integers and floats with the 1.2 core ones.
_CaseLoader.yaml_implicit_resolvers = {
    first: [
        (tag, regexp) for tag, regexp in resolvers if tag not in (_BOOL_TAG, _INT_TAG, _FLOAT_TAG)
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_CaseLoader.add_implicit_resolver(
    _BOOL_TAG, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_CaseLoader.add_implicit_resolver(_INT_TAG, re.compile(r"^[-+]?[0-9]+$"), list("-+0123456789"))
_CaseLoader.add_implicit_resolver(
    _FLOAT_TAG,
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)
# The 1.1 integer constructor reads a leading zero as octal; a decimal integer is what 1.2 means.
_CaseLoader.add_constructor(_INT_TAG, lambda loader, node: int(loader.construct_scalar(node)))


@dataclasses.dataclass(frozen=True)
class CheckedCase:
    """A case whose file and mesh have both been read, and which fits its mesh.

    Attributes
    ----------
    spec : CaseSpec
        The case.
    mesh : Mesh
        Its mesh, validated -- topology only; no geometry has been computed.
    """

    spec: CaseSpec
    mesh: Mesh

    def build(self) -> object:
        """The case's problem: its mesh's geometry, then its equations.

        This is where the cost of loading a case starts -- the geometry, and for a Reynolds-averaged
        case the wall distance. Its physics decides what is built (see
        :meth:`~aquaflux.case.Physics.build`).

        Returns
        -------
        MomentumContinuity or CoupledRANS
            The flow assembler for a laminar case; the coupled flow and closure for a
            Reynolds-averaged one.
        """
        return self.spec.physics.build(self.spec, self.mesh, self.mesh.geometry())

    def solve(self, problem: object, **observers: object) -> object:
        """Solve ``problem`` with the case's solver (see :meth:`~aquaflux.case.SolverSpec.solve`).

        Parameters
        ----------
        problem : object
            What :meth:`build` returned for this case.
        **observers
            Observers of the solve, passed to the library solve beside the case's settings; a keyword
            that is one of its settings is refused.

        Returns
        -------
        object
            The converged fields: ``(flow, k, omega)`` for a Reynolds-averaged case, the flow state for
            a laminar one.

        Raises
        ------
        ValueError
            If the case states no solver and its physics' default cannot solve it.
        TypeError
            If an observer keyword is one of the solve's settings.
        """
        return solver_for(self.spec).solve(problem, **observers)


@dataclasses.dataclass(frozen=True)
class CaseFile:
    """A case as read from a file, with the directory its relative paths are taken from.

    Attributes
    ----------
    spec : CaseSpec
        The case the file describes.
    directory : pathlib.Path
        The directory the file sits in; a relative mesh path is relative to it.
    """

    spec: CaseSpec
    directory: Path

    def check(self) -> CheckedCase:
        """Read the case's mesh, validate it, and check the case against it.

        Returns
        -------
        CheckedCase
            The case with its validated mesh.

        Raises
        ------
        ValueError
            If the mesh is not topologically valid, or the case's patches do not fit it
            (:meth:`CaseSpec.check_against`).
        FileNotFoundError
            If the mesh cannot be found.
        """
        mesh = self.spec.mesh.read(self.directory).validate()
        self.spec.check_against(mesh)
        return CheckedCase(spec=self.spec, mesh=mesh)


def read_case(path: str | Path) -> CaseFile:
    """Read a case file.

    Parameters
    ----------
    path : str or path-like
        The case file.

    Returns
    -------
    CaseFile
        The case, checked on its own terms; :meth:`CaseFile.check` checks it against its mesh.

    Raises
    ------
    ValueError
        If the file is not a mapping of sections, or a setting in it is refused (see
        :func:`~aquaflux.case.case_spec_from_mapping`). The message begins with the file's path.
    TypeError
        If a value is of the wrong family for its position.
    yaml.YAMLError
        If the file is not valid YAML, or a mapping in it names one key twice.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        document = yaml.load(stream, Loader=_CaseLoader)
    try:
        spec = case_spec_from_mapping(document)
    except (ValueError, TypeError) as error:
        raise type(error)(f"{path}: {error}") from error
    return CaseFile(spec=spec, directory=path.parent)


def write_case(spec: CaseSpec, path: str | Path) -> None:
    """Write a case file that :func:`read_case` reads back as an equal case.

    Parameters
    ----------
    spec : CaseSpec
        The case. A relative mesh path is written as it is, so it is relative to wherever the file is
        written.
    path : str or path-like
        The file to write; replaced if it exists.

    Raises
    ------
    TypeError
        If ``spec`` holds a value no case file can name (see
        :func:`~aquaflux.case.case_spec_to_mapping`).
    """
    mapping = case_spec_to_mapping(spec)
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(mapping, stream, sort_keys=False, default_flow_style=False)
