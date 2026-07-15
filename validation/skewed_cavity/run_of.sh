#!/bin/bash
# Runs inside the openfoam13 container (OpenFOAM env already sourced by the image
# entrypoint). Meshes the skewed cavity once, then solves it steady (SIMPLE) for
# a sweep of nNonOrthogonalCorrectors, leaving the ASCII polyMesh + converged
# fields on the host for the aquaflux comparison.
#
# Invoke from the host as:
#   docker run --rm -v <this dir>:/work -w /work openfoam13:latest bash run_of.sh
set -e

WORK=/work
cd "$WORK"

echo "==================================================================="
echo " Skewed lid-driven cavity (beta=45, Re=100) -- OpenFOAM 13"
echo "==================================================================="

rm -rf runs
mkdir -p runs

# --- mesh once -----------------------------------------------------------------
cp -r of_case runs/mesh
cd runs/mesh
echo ">>> blockMesh"
blockMesh > log.blockMesh 2>&1
echo ">>> checkMesh"
checkMesh > log.checkMesh 2>&1
echo ">>> writeCellCentres (for cell<->cell mapping into aquaflux)"
postProcess -func writeCellCentres -time 0 > log.cellCentres 2>&1
echo "--- checkMesh non-orthogonality ---"
grep -i "non-orthogonality" log.checkMesh || true
cd "$WORK"

# --- sweep nNonOrthogonalCorrectors -------------------------------------------
# nNonOrthogonalCorrectors is the number of deferred non-orthogonal correction
# sweeps in the segregated pressure solve. On this 45-degree mesh, 0 correctors
# diverges (the correction is absent); >=1 converges. Each case is capped with
# `timeout` so a diverging case cannot run away, and divergence is recorded
# rather than treated as a script failure.
for n in 0 1 2; do
    echo ""
    echo ">>> foamRun  nNonOrthogonalCorrectors = $n  (cap 150s)"
    RUN="runs/nonorth_$n"
    cp -r runs/mesh "$RUN"
    cd "$RUN"
    sed -i "s/nNonOrthogonalCorrectors [0-9]*;/nNonOrthogonalCorrectors $n;/" system/fvSolution
    timeout 150 foamRun > log.foamRun 2>&1 || true
    if grep -q "SIMPLE solution converged" log.foamRun; then
        ITERS=$(grep "SIMPLE solution converged" log.foamRun | grep -oE "[0-9]+ iterations" | grep -oE "[0-9]+")
        echo "    CONVERGED in $ITERS outer iterations"
    elif grep -q "nan\|inf" log.foamRun; then
        echo "    DIVERGED (nan/inf) -- the uncorrected non-orthogonal solve is unstable"
    else
        echo "    did not converge within the cap (latest time $(ls | grep -E '^[0-9]+$' | sort -n | tail -1))"
    fi
    cd "$WORK"
done

echo ""
echo "=== DONE. Runs under runs/ (mesh, nonorth_0, nonorth_1, nonorth_2) ==="
