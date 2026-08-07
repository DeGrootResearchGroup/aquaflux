#!/bin/bash
# Time-accurate OpenFOAM kOmegaSST reference for the 3D backward-facing step. The steady SIMPLE run
# limit-cycles on the separated 3D flow, so this transient run to a statistically-steady state is the
# comparison target (as for the 2D case). Run inside the openfoam13 container:
#   docker run --rm -v <study dir>:/work -w /work/of_transient openfoam13:latest bash run_transient.sh
#
# Pass --tight to re-run with `system/fvSolution.tight` into `runs/kwsst_transient_tight`, leaving the
# reference untouched. That variant exists to put a convergence bar on the reference itself: the
# default settings stop each timestep at 1% of its own residual, so the run is steady (Ux decays three
# decades by t = 0.15 and stays there) but converged only to ~1% per step -- not a firm enough baseline
# to attribute a reattachment difference to physics. Compare the two x_r/h values, not the fields.
#
# The mesh, physical properties and boundary conditions are the SAME as of_case -- this script copies
# them in rather than keeping a second copy, so the transient reference is byte-identical in geometry
# and setup to the steady case and to what aquaflux imports.
set -e
cd "$(dirname "$0")"
RUNS=../runs
if [ "${1:-}" = "--tight" ]; then
    DST="$RUNS/kwsst_transient_tight"
    SOLUTION=system/fvSolution.tight
else
    DST="$RUNS/kwsst_transient"
    SOLUTION=system/fvSolution
fi

# One geometry/BC/property definition, shared with the steady case.
cp ../of_case/system/blockMeshDict system/blockMeshDict
mkdir -p constant
cp ../of_case/constant/momentumTransport ../of_case/constant/physicalProperties constant/
rm -rf 0 [1-9]* 0.[0-9]* processor* postProcessing log.* 2>/dev/null || true
cp -r ../of_case/0.orig 0

# OpenFOAM reads `system/fvSolution` by name, so the chosen variant is swapped in and restored on the
# way out (including on failure) -- the tracked default must survive a tight run.
if [ "$SOLUTION" != system/fvSolution ]; then
    cp system/fvSolution system/fvSolution.default
    trap 'mv -f system/fvSolution.default system/fvSolution 2>/dev/null || true' EXIT
    cp "$SOLUTION" system/fvSolution
fi

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
