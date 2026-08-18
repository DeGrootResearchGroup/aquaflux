"""Write the OpenFOAM tracer field ``0/s``, with the inlet band generated from aquaflux's mesh.

The injection profile has one definition (``injector.injected_value``) and both codes must see the
*same* per-face values, or a difference in the injector shows up downstream as a difference in
transport. Rather than restate the profile as a coded boundary condition in the OpenFOAM case -- a
second implementation, in another language, that would have to be kept in step -- this evaluates the
one definition on the inlet face centroids and writes the result as an explicit
``nonuniform List<scalar>``.

That is only sound because the two codes agree on what face ``i`` of the inlet patch is, which is
the same index correspondence :func:`~aquaflux.io.read_surface_scalar_field` relies on and checks:
OpenFOAM writes a patch's values in its own face order, aquaflux's patch indices come back ascending
over a contiguous block, and the import never renumbers. The reader's structural guard runs here too
(it is called on the same mesh), so a case whose ordering did not correspond would raise rather than
write a plausible file.

Writes ``of_case/0.orig/s``. Run from the repo root before the OpenFOAM run::

    python3 validation/bfs3d_species/write_inlet_field.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

from aquaflux.io import read_openfoam  # noqa: E402
from injector import injected_value  # noqa: E402

#: The polyMesh both codes share. The species case deliberately does NOT carry its own mesh: it runs
#: on the flow case's converged fields, so it must run on the flow case's cells.
FLOW_CASE = ROOT / "validation" / "bfs3d_openfoam" / "of_case"

#: The transported field's name, matching the ``scalarTransport`` function object in ``system/``.
FIELD = "s"

_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volScalarField;
    object      {field};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// The inlet's values are GENERATED, not written by hand -- see write_inlet_field.py. They are the
// injection profile evaluated on this mesh's inlet face centroids, so this case and aquaflux impose
// the identical boundary values face for face.

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           nonuniform List<scalar>
{count}
(
{values}
)
;
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    upperWall
    {{
        type            zeroGradient;
    }}

    lowerWall
    {{
        type            zeroGradient;
    }}

    sideWalls
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""


def inlet_values(mesh, geometry) -> np.ndarray:
    """The injected value at each inlet face, in the patch's own ascending face order.

    Parameters
    ----------
    mesh : Mesh
        The imported mesh, whose ``face_patches`` carries the ``inlet`` patch.
    geometry : MeshGeometry
        Its geometry, for the face centroids the profile is a function of.

    Returns
    -------
    np.ndarray
        Imposed value per inlet face, shape ``(n_inlet_faces,)``.
    """
    indices = np.asarray(mesh.face_patches.indices("inlet"))
    centroids = np.asarray(geometry.face.centroid)[indices]
    return np.asarray(injected_value(centroids))


def main() -> int:
    mesh = read_openfoam(FLOW_CASE)
    geometry = mesh.geometry()
    values = inlet_values(mesh, geometry)

    # The same structural guard the flux reader applies, run here so a case whose face ordering did
    # not correspond fails at generation rather than producing a file that looks fine.
    from aquaflux.io import read_surface_scalar_field

    read_surface_scalar_field(FLOW_CASE / "2000" / "phi", mesh)

    target = CASE / "of_case" / "0.orig" / FIELD
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _HEADER.format(
            field=FIELD,
            count=values.size,
            values="\n".join(f"{v:.12g}" for v in values),
        )
    )
    print(f"wrote {target.relative_to(ROOT)}: {values.size} inlet faces")
    print(f"  injected value range [{values.min():.3f}, {values.max():.3f}]")
    print(f"  faces above half the peak: {int((values > 0.5 * values.max()).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
