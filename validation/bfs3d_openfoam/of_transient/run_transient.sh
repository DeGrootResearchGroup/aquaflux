#!/bin/bash
# Time-accurate OpenFOAM kOmegaSST reference for the 3D backward-facing step. The steady SIMPLE run
# limit-cycles on the separated 3D flow, so this transient run to a statistically-steady state is the
# comparison target (as for the 2D case). Run inside the openfoam13 container:
#   docker run --rm -v <study dir>:/work -w /work/of_transient openfoam13:latest bash run_transient.sh
#
# The mesh, physical properties and boundary conditions are the SAME as of_case -- this script copies
# them in rather than keeping a second copy, so the transient reference is byte-identical in geometry
# and setup to the steady case and to what aquaflux imports.
set -e
cd "$(dirname "$0")"
RUNS=../runs
DST="$RUNS/kwsst_transient"

# One geometry/BC/property definition, shared with the steady case.
cp ../of_case/system/blockMeshDict system/blockMeshDict
mkdir -p constant
cp ../of_case/constant/momentumTransport ../of_case/constant/physicalProperties constant/
rm -rf 0 [1-9]* 0.[0-9]* processor* postProcessing log.* 2>/dev/null || true
cp -r ../of_case/0.orig 0

blockMesh > log.blockMesh 2>&1
foamRun > log.foamRun 2>&1 || true

T=$(foamListTimes -latestTime)
mkdir -p "$DST"; rm -f "$DST"/*
# Prefer the time-averaged mean fields; fall back to the instantaneous latest field if the run went
# fully steady before averaging began (then UMean == U anyway).
for base in U p k omega nut; do
    if [ -f "$T/${base}Mean" ]; then
        cp "$T/${base}Mean" "$DST/$base"
    else
        cp "$T/$base" "$DST/$base"
    fi
done
cp "postProcessing/residuals(p,U,k,omega)/0/residuals.dat" "$DST/residuals.dat" 2>/dev/null || true

echo "bfs3d transient kOmegaSST: latestTime=$T (fields -> $DST, *Mean preferred)" | tee "$RUNS/summary_transient.txt"
