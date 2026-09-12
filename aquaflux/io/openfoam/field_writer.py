"""Write computed cell fields back into an OpenFOAM case as a time directory.

The counterpart of :mod:`.fields`, which reads one. A solved aquaflux state is written as an
ordinary OpenFOAM time directory, so the result can be opened by the same post-processing tools the
reference solution uses -- viewed beside that reference on the same mesh, or used as the restart
state for a continued run.

**Only the internal values are ours.** Everything else in a field file -- the dimensions, and the
per-patch ``boundaryField`` dictionary -- is copied through from a *template*: an existing field of
the same name in the same case, normally the one in the ``0`` directory the case was set up with.
That is what makes a written field genuinely restartable rather than merely readable. The boundary
conditions are then the case's own by construction, spelled the way its solver expects, including
the types that carry no value (``zeroGradient``, ``noSlip``, ``symmetry``) and the ones this package
cannot reconstruct at all: an ``empty`` patch is removed from the mesh by the two-dimensional
collapse, so a writer working from the imported mesh alone could not put it back.

**Why the internal block is a direct dump.** A cell's aquaflux index is OpenFOAM's own. The
assembler derives cell indices from the ``owner``/``neighbour`` labels it reads rather than
renumbering, and the ``empty``-patch collapse that turns an extruded mesh two-dimensional keeps
cells at their indices -- it removes faces, not cells. So cell ``i`` here is cell ``i`` there, and
the internal field needs no permutation. Faces are a different matter (the collapse does renumber
them), which is a second reason the patch dictionaries are copied rather than rebuilt.

⚠️ **A two-dimensional field is padded back to three components, and which axis was dropped cannot
be recovered from the mesh.** The collapse infers the extruded axis and does not record it, so
``extruded_axis`` says where the zero goes. The default is the last axis, which is the convention a
mesh extruded for a two-dimensional case is built with.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .foamfile import read_foam_body, read_foam_file, resolve_polymesh_dir
from .grammar import parse_vector_list, split_boundary_blocks

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from aquaflux.mesh import Mesh

_DIMENSIONS_RE = re.compile(r"\bdimensions\s+(\[[^\]]*\])\s*;")
_INTERNAL_FIELD_RE = re.compile(r"\binternalField\s+(.*?);", re.DOTALL)
#: The one macro a written field must resolve itself. Every other ``$name`` in a patch
#: dictionary still resolves in the written file, because the whole ``boundaryField`` and the
#: ``dimensions`` entry are carried across unchanged -- but ``internalField`` is precisely what
#: this writer replaces, so a reference to it would resolve against our values instead of the
#: template's.
_INTERNAL_FIELD_MACRO_RE = re.compile(r"\$\{?internalField\}?")

#: Components a field class carries. A class this does not name is refused rather than guessed at:
#: the component count decides how the values are laid out, so getting it wrong writes a file that
#: parses and means something else.
_CLASS_COMPONENTS = {"volScalarField": 1, "volVectorField": 3}

#: Significant digits per written value. Wide enough that a restart resumes from the state that was
#: written rather than from a rounding of it; ``repr``-grade round-tripping is not needed because
#: the consumer parses decimal text into doubles either way.
_PRECISION = 12

# The banner every file in the format carries. The editor mode marker the format normally puts
# on the first line is dropped: it is a syntax-highlighting hint for one editor, nothing reads
# it back, and it names a language this repository keeps out of its own sources.
_BANNER = r"""/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/"""

_SEPARATOR = "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"


class FieldTemplate(NamedTuple):
    """The parts of an existing field file a written field inherits.

    Attributes
    ----------
    class_name : str
        The ``FoamFile`` ``class`` entry -- ``volScalarField`` or ``volVectorField`` -- which fixes
        how many components each value carries.
    dimensions : str
        The ``dimensions`` entry verbatim, brackets included, e.g. ``[0 1 -1 0 0 0 0]``. Taken from
        the template rather than from a table here so there is no second place for a field's units
        to be recorded, and no way for the two to disagree.
    internal_field : str
        The template's own ``internalField`` entry (e.g. ``uniform 440.15``), which is **not**
        written -- the whole point is that the values replace it. It is kept because a patch
        dictionary may refer to it by macro: ``value $internalField;`` is the idiomatic way a ``0``
        directory gives an inlet the same value as the interior. That macro has to resolve against
        the template's internal field, since resolving it against the written one expands a whole
        cell-length list onto a patch.
    boundary : dict of {str: str}
        Patch name to its ``boundaryField`` dictionary body, in file order, copied through
        otherwise unchanged.
    """

    class_name: str
    dimensions: str
    internal_field: str
    boundary: dict[str, str]


def parse_field_template(body: str, header: Mapping[str, str]) -> FieldTemplate:
    """Build a :class:`FieldTemplate` from a parsed field file. Pure: tests on a snippet.

    Parameters
    ----------
    body : str
        The comment-stripped field-file body.
    header : mapping of {str: str}
        The file's ``FoamFile`` header entries.

    Returns
    -------
    FieldTemplate
        The class, dimensions and per-patch dictionaries to inherit.

    Raises
    ------
    ValueError
        If the header names no ``class``, the class is not a cell field this can write, or the body
        has no ``dimensions`` entry.
    """
    class_name = header.get("class")
    if class_name is None:
        raise ValueError("template has no 'class' entry in its FoamFile header")
    if class_name not in _CLASS_COMPONENTS:
        known = ", ".join(sorted(_CLASS_COMPONENTS))
        raise ValueError(
            f"template class '{class_name}' is not a cell field; expected one of {known}"
        )
    match = _DIMENSIONS_RE.search(body)
    if match is None:
        raise ValueError("template has no 'dimensions' entry")
    internal = _INTERNAL_FIELD_RE.search(body)
    return FieldTemplate(
        class_name,
        match.group(1),
        internal.group(1).strip() if internal else "",
        split_boundary_blocks(body),
    )


def read_field_template(path) -> FieldTemplate:
    """Read an existing field file and keep the parts a written field inherits.

    Parameters
    ----------
    path : str or Path
        The template field file, e.g. ``<case>/0/U``.

    Returns
    -------
    FieldTemplate
        Its class, dimensions and per-patch boundary dictionaries.

    Raises
    ------
    FileNotFoundError
        If no file is there.
    ValueError
        If it is binary, or is not a readable cell field.
    """
    foam = read_foam_file(Path(path))
    return parse_field_template(foam.body, foam.header)


def infer_extruded_axis(case, mesh: Mesh) -> int:
    """Which axis the two-dimensional collapse removed, recovered from the case's own polyMesh.

    The collapsed mesh cannot answer this: its ``node_coords`` really are ``(n_nodes, 2)``, so a
    case extruded along ``y`` and one extruded along ``z`` collapse to identical meshes. The source
    polyMesh still has three coordinates, and comparing extents identifies the missing one.

    **Exact rather than a heuristic.** The collapse keeps the surviving axes in *ascending* order
    and preserves their coordinate values, so there are only three candidates -- it dropped 0, 1 or
    2 -- and the extents of the two axes it kept must equal the collapsed mesh's own two extents.
    Picking "the thinnest axis" instead would be a guess, and would misread a domain that is
    genuinely thin in a resolved direction.

    ⚠️ Cheap **because of what needs it**: a mesh that was collapsed is one cell thick by
    construction, so it is never one of the large ones. A genuinely three-dimensional mesh never
    reaches here.

    Parameters
    ----------
    case : str or Path
        The case directory, or the polyMesh directory itself.
    mesh : Mesh
        The collapsed two-dimensional mesh.

    Returns
    -------
    int
        The axis (0, 1 or 2) the collapse removed.

    Raises
    ------
    ValueError
        If ``mesh`` is not two-dimensional, or if the extents match no candidate or more than one --
        in which case the caller should say which axis it was.
    """
    if mesh.dim != 2:
        raise ValueError(f"expected a 2D mesh to have been collapsed; got dim {mesh.dim}")
    points = parse_vector_list(read_foam_body(resolve_polymesh_dir(case) / "points"))
    spatial = np.ptp(points, axis=0)
    planar = np.ptp(np.asarray(mesh.node_coords, dtype=float), axis=0)
    scale = float(np.max(spatial)) or 1.0
    matched = [
        dropped
        for dropped in range(3)
        if np.allclose(
            [spatial[axis] for axis in range(3) if axis != dropped],
            planar,
            rtol=1e-9,
            atol=1e-9 * scale,
        )
    ]
    if len(matched) == 1:
        return matched[0]
    raise ValueError(
        f"cannot tell which axis the mesh was collapsed along: the polyMesh extents {spatial} "
        f"match {'no' if not matched else 'more than one'} candidate against the collapsed mesh's "
        f"{planar} (candidates {matched}); pass extruded_axis explicitly"
    )


def _components(values: np.ndarray, template: FieldTemplate, extruded_axis: int) -> np.ndarray:
    """Values as ``(n_cells, n_components)`` for the template's class, padding a 2D vector to 3."""
    wanted = _CLASS_COMPONENTS[template.class_name]
    if wanted == 1:
        if values.ndim != 1:
            raise ValueError(
                f"{template.class_name} expects values of shape (n_cells,); got {values.shape}"
            )
        return values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError(
            f"{template.class_name} expects values of shape (n_cells, 2) or (n_cells, 3); "
            f"got {values.shape}"
        )
    if values.shape[1] == 3:
        return values
    # Validated before it is normalized: `% 3` alone would turn a nonsense axis into a plausible
    # one and put the zero somewhere silently, which is the whole failure this argument invites.
    if not -3 <= extruded_axis <= 2:
        raise ValueError(f"extruded_axis must be one of -3..2; got {extruded_axis}")
    return np.insert(values, extruded_axis % 3, 0.0, axis=1)


def _resolve_internal_field(block: str, template: FieldTemplate, patch: str) -> str:
    """A patch dictionary with ``$internalField`` replaced by the template's own internal entry.

    ``value $internalField;`` is how a ``0`` directory gives a patch the same value as the interior,
    and it is only well defined while that interior value is uniform. This writer always writes a
    list, so carrying the macro through would expand every cell onto the patch -- which OpenFOAM
    rejects, reporting a transferred compound at the patch's ``value`` rather than anything naming
    the macro. Resolving it against the template restores what the entry meant where it was written.
    """
    if not _INTERNAL_FIELD_MACRO_RE.search(block):
        return block
    if not template.internal_field:
        raise ValueError(
            f"patch '{patch}' refers to $internalField but the template has no internalField entry"
        )
    return _INTERNAL_FIELD_MACRO_RE.sub(template.internal_field.replace("\\", "\\\\"), block)


def _reindent_block(block: str, indent: str) -> list[str]:
    """A copied patch dictionary's lines, re-indented to ``indent`` and keeping their own nesting.

    Two shapes have to come out right, and they pull in opposite directions. The block's first line
    sits on the same line as its opening brace, so it carries no indent of its own; and a written
    ``value nonuniform List<scalar>`` puts its count, parentheses and entries at **column zero**,
    below entries indented normally. Taking the smallest indent in the block would therefore measure
    the list and dedent nothing, while stripping each line would flatten the nesting. So the
    reference is the first line that *has* an indent to speak for the block -- the second -- and
    lines shallower than it simply land at ``indent``.
    """
    lines = block.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    measured = [line for line in lines[1:] if line.strip()]
    reference = len(measured[0]) - len(measured[0].lstrip()) if measured else 0
    return [
        indent + line[min(reference, len(line) - len(line.lstrip())) :] if line.strip() else ""
        for line in lines
    ]


def _format_entry(row: np.ndarray) -> str:
    """One value: a bare number for a scalar, a parenthesized tuple for a vector."""
    parts = [f"{v:.{_PRECISION}g}" for v in row]
    return parts[0] if len(parts) == 1 else "(" + " ".join(parts) + ")"


def format_volume_field(
    values,
    template: FieldTemplate,
    *,
    object_name: str,
    location: str,
    extruded_axis: int = -1,
    allow_non_finite: bool = False,
) -> str:
    """Serialize cell values into an OpenFOAM field file. Pure: no filesystem.

    Parameters
    ----------
    values : array-like
        Cell values, shape ``(n_cells,)`` for a scalar field or ``(n_cells, 2 | 3)`` for a vector
        one. Converted with ``np.asarray``, so a JAX array is accepted.
    template : FieldTemplate
        The class, dimensions and boundary dictionaries to inherit.
    object_name : str
        The field's name, written as the ``object`` header entry -- and the file name it must be
        saved under, since that is how the reading solver finds it.
    location : str
        The time directory the file will sit in, written as the ``location`` header entry.
    extruded_axis : int, optional
        Which of the three axes the two-dimensional collapse removed, so a ``(n_cells, 2)`` vector
        is padded with a zero there. Default ``-1`` (the last axis). Ignored for a scalar field and
        for values that already carry three components.
    allow_non_finite : bool, optional
        Permit ``NaN``/``inf`` values. Default ``False``, because the point of writing a field is
        that it can be read back, and a solver handed a non-finite restart state fails in a way that
        does not name this file. Set it to inspect a diverged state.

    Returns
    -------
    str
        The complete file text.

    Raises
    ------
    ValueError
        If the values' shape does not match the template's class, or they hold non-finite entries
        and ``allow_non_finite`` is not set.
    """
    array = np.asarray(values, dtype=float)
    if not allow_non_finite and not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"field '{object_name}' holds {bad} non-finite value(s), so a solver reading it back "
            f"would fail without naming this file; pass allow_non_finite=True to write it anyway"
        )
    rows = _components(array, template, extruded_axis)

    lines = [
        _BANNER,
        "FoamFile",
        "{",
        "    format      ascii;",
        f"    class       {template.class_name};",
        f'    location    "{location}";',
        f"    object      {object_name};",
        "}",
        _SEPARATOR,
        "",
        f"dimensions      {template.dimensions};",
        "",
        # Always the list form, never `uniform`: a computed field is not uniform, and one spelling
        # keeps the writer's output the same shape whatever the values happen to be.
        f"internalField   nonuniform List<{'scalar' if rows.shape[1] == 1 else 'vector'}>",
        f"{rows.shape[0]}",
        "(",
    ]
    lines.extend(_format_entry(row) for row in rows)
    lines.extend([")", ";", "", "boundaryField", "{"])
    for name, block in template.boundary.items():
        lines.append(f"    {name}")
        lines.append("    {")
        lines.extend(_reindent_block(_resolve_internal_field(block, template, name), "        "))
        lines.append("    }")
    lines.extend(["}", "", _SEPARATOR, ""])
    return "\n".join(lines)


def write_openfoam_field(
    path,
    values,
    *,
    template,
    object_name: str | None = None,
    extruded_axis: int = -1,
    allow_non_finite: bool = False,
) -> Path:
    """Write one cell field to ``path``, inheriting everything but the values from ``template``.

    Parameters
    ----------
    path : str or Path
        Destination file, e.g. ``<case>/2500/U``. Its parent directory is created if absent.
    values : array-like
        Cell values, as in :func:`format_volume_field`.
    template : FieldTemplate or str or Path
        The field to inherit dimensions and boundary conditions from -- a template already read, or
        a path to read one from.
    object_name : str, optional
        The field name written into the header. Defaults to ``path``'s file name, which is the name
        a solver will look the field up under.
    extruded_axis, allow_non_finite
        As in :func:`format_volume_field`.

    Returns
    -------
    Path
        The file written.
    """
    destination = Path(path)
    resolved = template if isinstance(template, FieldTemplate) else read_field_template(template)
    text = format_volume_field(
        values,
        resolved,
        object_name=object_name or destination.name,
        location=destination.parent.name,
        extruded_axis=extruded_axis,
        allow_non_finite=allow_non_finite,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def write_openfoam_time(
    case,
    time,
    fields: Mapping[str, object],
    mesh: Mesh,
    *,
    template_time: str = "0",
    extruded_axis: int | None = None,
    allow_non_finite: bool = False,
) -> Path:
    """Write a set of cell fields as one OpenFOAM time directory of ``case``.

    Each field inherits its dimensions and boundary conditions from the file of the same name in
    the case's ``template_time`` directory, so the written directory is a valid restart state for
    the solver that case is set up for.

    Parameters
    ----------
    case : str or Path
        The OpenFOAM case directory -- the one holding ``constant/polyMesh`` and the time
        directories, i.e. what :func:`~aquaflux.io.read_openfoam` was pointed at.
    time : str or float
        The time directory to write, e.g. ``2500`` or ``"2500"``. Used verbatim as the directory
        name when a string, so a case whose times are written a particular way keeps that spelling.
    fields : mapping of {str: array-like}
        Field name to cell values. The name is both the file written and the template looked up.
    mesh : Mesh
        The mesh the values live on. Used to check each field's length against ``n_cells`` -- the
        one thing a template cannot check, and the error worth catching, since a field of the wrong
        length parses as a valid file that no solver can read.
    template_time : str, optional
        The time directory to take dimensions and boundary conditions from. Default ``"0"``.
    extruded_axis : int, optional
        Which axis the two-dimensional collapse removed, so a ``(n_cells, 2)`` vector is padded
        with a zero there. Default ``None`` recovers it from the case's own polyMesh
        (:func:`infer_extruded_axis`) rather than assuming a convention -- and only when a field
        actually needs it, so a scalar-only or three-dimensional write reads no points at all.
        Pass an int to override, which is also the answer if the extents cannot distinguish the
        axes.
    allow_non_finite
        As in :func:`format_volume_field`.

    Returns
    -------
    Path
        The time directory written.

    Raises
    ------
    FileNotFoundError
        If the template directory, or a template for one of the named fields, is missing.
    ValueError
        If a field's length is not ``mesh.n_cells``.
    """
    case_dir = Path(case)
    template_dir = case_dir / str(template_time)
    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"no template time directory '{template_dir}'; a written field inherits its dimensions "
            f"and boundary conditions from the case's own field of the same name"
        )
    destination = case_dir / str(time)
    prepared = []
    for name, values in fields.items():
        array = np.asarray(values, dtype=float)
        if array.shape[0] != mesh.n_cells:
            raise ValueError(
                f"field '{name}' has {array.shape[0]} values but the mesh has {mesh.n_cells} cells"
            )
        template_path = template_dir / name
        if not template_path.is_file():
            raise FileNotFoundError(
                f"no template '{template_path}' for field '{name}'; add one, or write the field "
                f"with write_openfoam_field and a template of your choosing"
            )
        prepared.append((name, array, read_field_template(template_path)))

    # Recovered only where it is actually needed -- a vector handed over short a component, on a
    # collapsed mesh. A scalar-only or three-dimensional write never opens the points file.
    axis = extruded_axis
    if axis is None and any(
        template.class_name == "volVectorField" and array.ndim == 2 and array.shape[1] == 2
        for _, array, template in prepared
    ):
        axis = infer_extruded_axis(case_dir, mesh)

    for name, array, template in prepared:
        write_openfoam_field(
            destination / name,
            array,
            template=template,
            object_name=name,
            extruded_axis=-1 if axis is None else axis,
            allow_non_finite=allow_non_finite,
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination
