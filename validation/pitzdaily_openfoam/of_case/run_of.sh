#!/bin/bash
# OpenFOAM kOmegaSST reference for the pitzDaily backward-facing step (steady, incompressible).
# Runs inside the openfoam13 container and writes the converged fields (+ cell centres), the mesh,
# and the SIMPLE residual history to ../runs/:
#   docker run --rm -v <study dir>:/work -w /work/of_case openfoam13:latest bash run_of.sh
#
# This is the shipped OpenFOAM pitzDailySteady tutorial with one change: the RAS model switched from
# the shipped kEpsilon to kOmegaSST (constant/momentumTransport), the closure aquaflux implements.
# U_in = 10 m/s, nu = 1e-5 -> Re ~ 25000 on the 25.4 mm inlet height; ~12225 cells, wall-function
# near-wall treatment.
set -e
cd "$(dirname "$0")"
RUNS=../runs
DST="$RUNS/kwsst"
mkdir -p "$DST"

rm -rf 0 [1-9]* processor* postProcessing log.* 2>/dev/null || true
cp -r 0.orig 0
blockMesh > log.blockMesh 2>&1
foamRun > log.foamRun 2>&1 || true
foamPostProcess -func writeCellCentres -latestTime > log.cc 2>&1 || true

T=$(foamListTimes -latestTime)
rm -rf "$DST"; mkdir -p "$DST"
cp "$T"/{Ccx,Ccy,U,p,k,omega,nut} "$DST"/
# The mesh, so aquaflux reads the *same* cells (a cell-for-cell comparison, not a matched-setup one).
cp -r constant/polyMesh "$DST"/polyMesh
# The SIMPLE residual convergence history (the `residuals` function object writes one column per field).
cp "postProcessing/residuals(p,U,k,omega)/0/residuals.dat" "$DST/residuals.dat"

echo "pitzDaily kOmegaSST: latestTime=$T  $(grep -o 'SIMPLE solution converged in [0-9]* iterations' log.foamRun || echo 'ran to endTime')" | tee "$RUNS/summary.txt"
