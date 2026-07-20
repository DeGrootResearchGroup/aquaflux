"""Unit tests for the uniform-shape padding that lets ``shard_map`` map over partitions.

Padding is operator-independent bookkeeping, so it is tested here without any devices and without
``shard_map``: the sizes are uniform, the real rows survive untouched, and — the property the whole
scheme rests on — the padding is **inert**, so a residual assembled on a padded local mesh agrees
with the same residual on the unpadded one, row for row.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.mesh import structured_grid_3d
from aquaflux.parallel import BlockPartitioner, PaddedLayout, pad_partition, partition_mesh
from aquaflux.properties import Constant, PropertyModel

N_PARTS = 4
GAMMA = 1.7
BOUNDARY = {
    "left": Dirichlet(1.0),
    "right": Dirichlet(0.0),
    "bottom": Neumann(0.25),
    "top": Neumann(-0.4),
    "back": ZeroGradient(),
    "front": Dirichlet(0.1),
}


@pytest.fixture(scope="module")
def decomposition():
    """A 4-way decomposition of a small 3D grid, with its global geometry."""
    mesh = structured_grid_3d(5, 5, 5, named_boundaries=True)
    geometry = mesh.geometry()
    pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, N_PARTS))
    return mesh, geometry, pmesh


@pytest.fixture(scope="module")
def layout(decomposition):
    _, _, pmesh = decomposition
    return PaddedLayout.from_partitioned(pmesh)


def _padded(decomposition, layout, p):
    _, geometry, pmesh = decomposition
    part = pmesh.partitions[p]
    return pad_partition(layout, p, part.mesh, part.local_geometry(geometry))


def test_layout_sizes_are_the_maxima_over_partitions(decomposition, layout):
    """The padded sizes bound every partition, and the null slots sit past all real data."""
    _, _, pmesh = decomposition
    assert layout.n_partitions == N_PARTS
    assert layout.n_owned_max == max(p.n_owned for p in pmesh.partitions)
    assert layout.n_ghost_max == max(p.n_ghost for p in pmesh.partitions)
    assert layout.n_faces_max == max(int(p.mesh.n_faces) for p in pmesh.partitions)
    # The null cell/face are past every partition's real cells/faces, on every partition.
    for p, part in enumerate(pmesh.partitions):
        assert layout.null_cell >= part.n_owned + part.n_ghost
        assert layout.null_face >= layout.n_faces[p]


def test_remap_cells_keeps_owned_and_shifts_ghosts(decomposition, layout):
    """Owned cells keep their index; ghosts shift to the start of the padded ghost block."""
    _, _, pmesh = decomposition
    for p, part in enumerate(pmesh.partitions):
        local = np.arange(part.mesh.n_cells)
        remapped = layout.remap_cells(local, p)
        np.testing.assert_array_equal(remapped[: part.n_owned], local[: part.n_owned])
        expected_ghosts = layout.n_owned_max + np.arange(part.n_ghost)
        np.testing.assert_array_equal(remapped[part.n_owned :], expected_ghosts)


def test_owned_scatter_gather_round_trips(decomposition, layout):
    """Gathering a global field to the padded owned layout and scattering back is the identity."""
    mesh, _, _ = decomposition
    field = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
    round_tripped = layout.scatter_owned_to_global(layout.owned_states_from_global(field))
    np.testing.assert_allclose(np.asarray(round_tripped), np.asarray(field))


def test_padded_shapes_are_uniform_across_partitions(decomposition, layout):
    """Every partition pads to identical cell, face, face-node and node array shapes."""
    shapes = set()
    for p in range(N_PARTS):
        padded_mesh, padded_geometry = _padded(decomposition, layout, p)
        shapes.add(
            (
                padded_mesh.n_cells,
                padded_mesh.n_faces,
                padded_mesh.n_nodes,
                int(padded_mesh.face_nodes.face_node_indices.shape[0]),
                int(padded_geometry.cell.volume.shape[0]),
                int(padded_geometry.face.area.shape[0]),
            )
        )
    assert len(shapes) == 1


def test_padded_nodes_preserve_real_coordinates(decomposition, layout):
    """Padding the node array appends only copies of node 0, leaving the real nodes untouched.

    The padded node count is the per-partition maximum, far below the global node count — the whole
    point of the partition-local node set, so the stacked sharded arrays do not replicate the global
    nodes on every device.
    """
    mesh, _, pmesh = decomposition
    assert layout.n_nodes_max < mesh.n_nodes  # far below full replication of the global node set
    for p, part in enumerate(pmesh.partitions):
        padded_mesh, _ = _padded(decomposition, layout, p)
        real = np.asarray(part.mesh.node_coords)
        padded = np.asarray(padded_mesh.node_coords)
        assert padded.shape == (layout.n_nodes_max, real.shape[1])
        np.testing.assert_array_equal(padded[: real.shape[0]], real)
        assert np.all(padded[real.shape[0] :] == real[0])  # padding rows are copies of node 0


def test_real_geometry_survives_padding_unchanged(decomposition, layout):
    """Real cells and faces keep their gathered geometry exactly — padding only appends.

    Cell *volumes* are checked explicitly: padding cells carry a benign unit volume, and a bug
    that gave every cell unit volume would leave the residual correct for diffusion (which never
    reads volume) but silently wrong for any transient or volume-source term.
    """
    _, geometry, pmesh = decomposition
    for p, part in enumerate(pmesh.partitions):
        local_geometry = part.local_geometry(geometry)
        _, padded_geometry = _padded(decomposition, layout, p)
        n_real_cells, n_real_faces = part.mesh.n_cells, int(part.mesh.n_faces)
        real_cells = layout.remap_cells(np.arange(n_real_cells), p)

        np.testing.assert_allclose(
            np.asarray(padded_geometry.cell.volume)[real_cells],
            np.asarray(local_geometry.cell.volume),
        )
        np.testing.assert_allclose(
            np.asarray(padded_geometry.cell.centroid)[real_cells],
            np.asarray(local_geometry.cell.centroid),
        )
        for name in ("area", "centroid", "normal"):
            np.testing.assert_allclose(
                np.asarray(getattr(padded_geometry.face, name))[:n_real_faces],
                np.asarray(getattr(local_geometry.face, name)),
            )


def test_padding_is_inert_and_finite(decomposition, layout):
    """Padding faces carry zero area, are owned by the null cell, and no geometry is degenerate."""
    _, _, pmesh = decomposition
    for p, part in enumerate(pmesh.partitions):
        padded_mesh, padded_geometry = _padded(decomposition, layout, p)
        n_real_faces = int(part.mesh.n_faces)
        pad = slice(n_real_faces, None)

        np.testing.assert_array_equal(np.asarray(padded_geometry.face.area)[pad], 0.0)
        # A padding face can only ever scatter into the null cell, never a real owned row.
        np.testing.assert_array_equal(
            np.asarray(padded_mesh.face_cells.owner)[pad], layout.null_cell
        )
        np.testing.assert_array_equal(np.asarray(padded_mesh.face_cells.neighbour)[pad], -1)
        # Nothing degenerate: a NaN in a discarded row still poisons the shared reduction's tangent.
        assert np.all(np.isfinite(np.asarray(padded_geometry.cell.centroid)))
        assert np.all(np.isfinite(np.asarray(padded_geometry.face.centroid)))
        assert np.all(np.asarray(padded_geometry.cell.volume) > 0.0)


def test_padding_does_not_change_the_residual(decomposition, layout):
    """The residual on a padded local mesh matches the unpadded one on every real cell.

    This is the property the whole padding scheme rests on, and it holds without ``shard_map`` or
    multiple devices — so a padding regression is caught by a plain unit test.
    """
    _, geometry, pmesh = decomposition
    properties = PropertyModel({"diffusivity": Constant(GAMMA)})
    boundary = BoundaryConditions(BOUNDARY)
    rng = np.random.default_rng(3)

    for p, part in enumerate(pmesh.partitions):
        local_geometry = part.local_geometry(geometry)
        padded_mesh, padded_geometry = _padded(decomposition, layout, p)

        unpadded = ResidualAssembler.build(
            part.mesh, local_geometry, properties, (DiffusionFlux(),), boundary
        )
        padded = ResidualAssembler.build(
            padded_mesh, padded_geometry, properties, (DiffusionFlux(),), boundary
        )

        phi_local = jnp.asarray(rng.standard_normal(part.mesh.n_cells))
        phi_padded = (
            jnp.zeros(padded_mesh.n_cells)
            .at[layout.remap_cells(np.arange(part.mesh.n_cells), p)]
            .set(phi_local)
        )

        r_unpadded = unpadded.residual(phi_local)
        r_padded = padded.residual(phi_padded)
        real_cells = layout.remap_cells(np.arange(part.mesh.n_cells), p)
        np.testing.assert_allclose(
            np.asarray(r_padded)[real_cells], np.asarray(r_unpadded), atol=1e-12
        )
