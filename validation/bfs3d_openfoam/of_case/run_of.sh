#!/bin/bash
# OpenFOAM kOmegaSST reference for the finite-width 3D backward-facing step (steady, incompressible).
# Runs inside the openfoam13 container and writes the converged fields (+ 3D cell centres), the mesh,
# and the SIMPLE residual history to ../runs/:
#   docker run --rm -v <study dir>:/work -w /work/of_case openfoam13:latest bash run_of.sh
#
# Genuinely 3D: step height h = 0.01 m, expansion ratio 2, span 4h between two no-slip side walls
# (no `empty` patches -- the mesh stays fully three-dimensional). U_in = 10 m/s, nu = 1e-5 ->
# Re_h = 10000. Wall-function near-wall treatment.
set -e
cd "$(dirname "$0")"
RUNS=../runs
DST="$RUNS/kwsst"
mkdir -p "$DST"

rm -rf 0 [1-9]* processor* postProcessing log.* 2>/dev/null || true
cp -r 0.orig 0
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1 || true
foamRun > log.foamRun 2>&1 || true
foamPostProcess -func writeCellCentres -latestTime > log.cc 2>&1 || true

T=$(foamListTimes -latestTime)
rm -rf "$DST"; mkdir -p "$DST"
# The 3D case needs the z cell-centre (Ccz) as well as Ccx/Ccy.
cp "$T"/{Ccx,Ccy,Ccz,U,p,k,omega,nut} "$DST"/
# The mesh, so aquaflux reads the *same* cells (a cell-for-cell comparison, not a matched-setup one).
cp -r constant/polyMesh "$DST"/polyMesh
# The SIMPLE residual convergence history (the `residuals` function object writes one column per field).
cp "postProcessing/residuals(p,U,k,omega)/0/residuals.dat" "$DST/residuals.dat" 2>/dev/null || true

echo "bfs3d kOmegaSST: latestTime=$T  $(grep -o 'SIMPLE solution converged in [0-9]* iterations' log.foamRun || echo 'ran to endTime')" | tee "$RUNS/summary.txt"
