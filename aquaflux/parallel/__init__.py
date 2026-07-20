"""Distributed-memory parallelization: domain decomposition and halo exchange.

Separates the parallelization concern from mesh storage: `Mesh` stays a pure
contiguous-block container, and this package owns the decomposition — the
`PartitionedMesh` wrapper (owned cells + a halo ring per partition) and the halo exchange that
refreshes ghost values across partitions. Build-time decomposition is pure numpy; the halo
exchange is the only new operator inside the differentiable, sharded residual.
"""

from __future__ import annotations

from .distributed import (
    PaddedDiffusion,
    build_padded_diffusion,
    distributed_diffusion_residual,
)
from .halo import AllGatherHaloExchange, HaloExchange
from .partition import LocalPartition, PartitionedMesh, partition_mesh
from .partitioner import (
    BlockPartitioner,
    Partitioner,
    ScotchCLIPartitioner,
    ScotchPartitioner,
)

__all__ = [
    "AllGatherHaloExchange",
    "BlockPartitioner",
    "HaloExchange",
    "LocalPartition",
    "PaddedDiffusion",
    "PartitionedMesh",
    "Partitioner",
    "ScotchCLIPartitioner",
    "ScotchPartitioner",
    "build_padded_diffusion",
    "distributed_diffusion_residual",
    "partition_mesh",
]
