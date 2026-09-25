"""What a mesh file declares about its patches -- a type each, and named groups of them -- survives import.

An OpenFOAM ``boundary`` file gives each patch a ``type`` and may put it in groups with ``inGroups``, so
that one name addresses every wall. The mesh carries both, uninterpreted: through the reader, the
two-dimensional collapse (which removes the capping patches and so must drop them from both records),
and the distributed partition and padding paths, which build their face patches directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import numpy as np
import pytest
from aquaflux.io import read_openfoam
from aquaflux.io.openfoam.grammar import parse_boundary
from aquaflux.mesh import Mesh, structured_grid_2d
from aquaflux.mesh.groups import FacePatches
from aquaflux.parallel import BlockPartitioner, PaddedLayout, pad_partition, partition_mesh

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _boundary_body(*entries: str) -> str:
    """A ``boundary`` file body holding one patch per entry, each entry the patch's extra lines."""
    blocks = "".join(
        f"p{i} {{ type wall; {extra} nFaces 1; startFace {i}; }}\n"
        for i, extra in enumerate(entries)
    )
    return f"{len(entries)}\n(\n{blocks})\n"


# --- reading inGroups --------------------------------------------------------------------------


def test_every_form_of_an_in_groups_entry_is_read_and_an_absent_one_is_empty() -> None:
    patches = parse_boundary(
        _boundary_body(
            "inGroups List<word> 2(wall heated);",
            "inGroups 1(wall);",
            "inGroups (wall heated);",
            "",
        )
    )
    assert [p.in_groups for p in patches] == [
        ("wall", "heated"),
        ("wall",),
        ("wall", "heated"),
        (),
    ]


def test_an_in_groups_count_that_disagrees_with_its_list_is_refused_naming_the_patch() -> None:
    with pytest.raises(
        ValueError, match=r"boundary patch 'p0' inGroups declares 3 entries but lists 2"
    ):
        parse_boundary(_boundary_body("inGroups List<word> 3(wall heated);"))


def _slab_with_groups(tmp_path: Path, source: str, groups: dict[str, str]) -> Path:
    """A copy of a slab fixture whose ``boundary`` file puts each named patch in the given groups."""
    mesh = tmp_path / "polyMesh"
    shutil.copytree(FIXTURES / source, mesh)
    text = (mesh / "boundary").read_text()
    for patch, entry in groups.items():
        text = text.replace(
            f"    {patch}\n    {{\n", f"    {patch}\n    {{\n        inGroups {entry};\n"
        )
    (mesh / "boundary").write_text(text)
    return mesh


def test_a_read_mesh_reports_each_patch_type_and_group_and_the_collapse_drops_the_caps(
    tmp_path: Path,
) -> None:
    """The slab is read as 2D, so its two ``empty`` caps are removed: their type goes, their own group
    goes with them, and a group they shared with a surviving patch keeps that patch alone."""
    directory = _slab_with_groups(
        tmp_path,
        "polymesh_2d_slab",
        {
            "top": "List<word> 2(wall lid)",
            "bottom": "1(wall)",
            "left": "1(ends)",
            "right": "1(ends)",
            "back": "2(empty lid)",
            "front": "1(empty)",
        },
    )
    patches = read_openfoam(directory).face_patches
    assert patches.patch_types == (
        ("left", "patch"),
        ("right", "patch"),
        ("bottom", "wall"),
        ("top", "wall"),
    )
    # Groups in the order the file first names them; members in patch order (bottom before top).
    assert patches.patch_groups == (
        ("ends", ("left", "right")),
        ("wall", ("bottom", "top")),
        ("lid", ("top",)),
    )


def test_a_three_dimensional_mesh_keeps_every_patch_type() -> None:
    patches = read_openfoam(FIXTURES / "polymesh_3d_two_cubes").face_patches
    named = [name for name in patches.names if name not in ("interior", "boundary")]
    assert [patch for patch, _ in patches.patch_types] == named
    assert all(kind for _, kind in patches.patch_types)


# --- the record on the mesh ----------------------------------------------------------------------


def _square() -> Mesh:
    return structured_grid_2d(2, 2, named_boundaries=True)


def _patches(**record: object) -> FacePatches:
    face_patches = _square().face_patches
    return FacePatches(label=face_patches.label, names=face_patches.names, **record)


def test_a_name_addresses_a_patch_or_the_members_of_a_group() -> None:
    patches = _patches(
        patch_groups=(("walls", ("bottom", "top")), ("left", ("left",))),
    )
    assert patches.addressed_by("right") == ("right",)
    assert patches.addressed_by("walls") == ("bottom", "top")
    # A group holding only the patch of the same name means that patch: nothing is ambiguous.
    assert patches.addressed_by("left") == ("left",)
    assert patches.group_members("walls") == ("bottom", "top")
    assert patches.group_names == ("walls", "left")


def test_a_name_that_is_a_patch_and_a_group_of_other_patches_is_refused() -> None:
    patches = _patches(patch_groups=(("top", ("bottom", "top")),))
    with pytest.raises(
        ValueError, match=r"'top' is both a patch and a patch group of \['bottom', 'top'\]"
    ):
        patches.addressed_by("top")


def test_a_name_that_is_neither_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no patch or patch group named 'lid'"):
        _patches().addressed_by("lid")


def test_a_patch_type_is_what_was_declared_or_none() -> None:
    patches = _patches(patch_types=(("bottom", "wall"),))
    assert patches.type_of("bottom") == "wall"
    assert patches.type_of("top") is None
    with pytest.raises(ValueError, match=r"no group named 'lid'"):
        patches.type_of("lid")


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"patch_types": (("lid", "wall"),)}, r"a type is given for \['lid'\]"),
        (
            {"patch_types": (("top", "wall"), ("top", "patch"))},
            r"a patch is given more than one type",
        ),
        (
            {"patch_groups": (("walls", ("top", "lid")),)},
            r"'walls' must list distinct named patches",
        ),
        (
            {"patch_groups": (("walls", ("top", "top")),)},
            r"'walls' must list distinct named patches",
        ),
        ({"patch_groups": (("walls", ()),)}, r"'walls' must list distinct named patches"),
        (
            {"patch_groups": (("walls", ("interior",)),)},
            r"'walls' must list distinct named patches",
        ),
        (
            {"patch_groups": (("boundary", ("top",)),)},
            r"patch group name\(s\) \['boundary'\] are reserved",
        ),
        (
            {"patch_groups": (("walls", ("top",)), ("walls", ("bottom",)))},
            r"a patch group is named twice",
        ),
    ],
    ids=[
        "type-of-no-patch",
        "two-types",
        "member-no-patch",
        "member-twice",
        "empty-group",
        "reserved-member",
        "reserved-group",
        "group-twice",
    ],
)
def test_a_record_that_does_not_describe_the_patches_is_refused(record, match) -> None:
    with pytest.raises(ValueError, match=match):
        _patches(**record)


def test_removing_patches_takes_them_out_of_both_records_and_drops_an_emptied_group() -> None:
    patches = _patches(
        patch_types=(("top", "wall"), ("bottom", "wall"), ("left", "patch")),
        patch_groups=(("walls", ("bottom", "top")), ("caps", ("top",))),
    )
    types, groups = patches.without(["top"])
    assert types == {"bottom": "wall", "left": "patch"}
    assert groups == {"walls": ("bottom",)}


def test_a_mesh_built_from_faces_carries_the_types_and_groups_it_is_given() -> None:
    #   one square cell between four boundary edges
    mesh = Mesh.from_faces(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        owner=[0, 0, 0, 0],
        neighbour=[-1, -1, -1, -1],
        n_cells=1,
        face_patches={"south": [0], "east": [1], "north": [2], "west": [3]},
        patch_types={"north": "wall", "south": "wall"},
        patch_groups={"walls": ["north", "south"]},
    )
    assert mesh.face_patches.patch_types == (("north", "wall"), ("south", "wall"))
    assert mesh.face_patches.patch_groups == (("walls", ("south", "north")),)


# --- the paths that build face patches directly ---------------------------------------------------


def _declared_square() -> Mesh:
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    declared = FacePatches(
        label=mesh.face_patches.label,
        names=mesh.face_patches.names,
        patch_types=(("bottom", "wall"), ("top", "wall")),
        patch_groups=(("walls", ("bottom", "top")),),
    )
    return eqx.tree_at(lambda m: m.face_patches, mesh, declared)


def test_a_partitioned_and_padded_mesh_keeps_the_declared_record() -> None:
    mesh = _declared_square()
    geometry = mesh.geometry()
    pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 2))
    layout = PaddedLayout.from_partitioned(pmesh)
    for p, part in enumerate(pmesh.partitions):
        local = part.mesh.face_patches
        assert (local.patch_types, local.patch_groups) == (
            mesh.face_patches.patch_types,
            mesh.face_patches.patch_groups,
        )
        padded, _ = pad_partition(layout, p, part.mesh, part.local_geometry(geometry))
        assert padded.face_patches.patch_types == mesh.face_patches.patch_types
        assert padded.face_patches.patch_groups == mesh.face_patches.patch_groups


def test_a_partitioned_mesh_built_without_a_record_still_constructs() -> None:
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 2))
    assert all(part.mesh.face_patches.patch_groups == () for part in pmesh.partitions)
