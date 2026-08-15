#!/bin/bash
# Mesh + solve ONE skewed-cavity beta case inside the openfoam13 container.
# The host (sweep.py) has already written <betadir>/case with the beta-specific
# blockMeshDict. Usage (from the skewed_cavity dir mounted at /work):
#   docker run --rm -v <skewed_cavity>:/work -w /work openfoam13 bash run_of_beta.sh <betadir>
set -e
BETADIR="$1"
cd "/work/$BETADIR"

cd case
blockMesh   > log.blockMesh 2>&1
checkMesh   > log.checkMesh 2>&1
postProcess -func writeCellCentres -time 0 > log.cc 2>&1
grep -i "non-orthogonality" log.checkMesh || true
cd ..

for n in 0 1 2; do
    rm -rf "nonorth_$n"
    cp -r case "nonorth_$n"
    cd "nonorth_$n"
    sed -i "s/nNonOrthogonalCorrectors [0-9]*;/nNonOrthogonalCorrectors $n;/" system/fvSolution
    timeout 120 foamRun > log.foamRun 2>&1 || true
    if grep -q "SIMPLE solution converged" log.foamRun; then
        echo "  nonorth=$n CONVERGED in $(grep 'SIMPLE solution converged' log.foamRun | grep -oE '[0-9]+ iterations' | grep -oE '[0-9]+')"
    elif grep -q "nan\|inf" log.foamRun; then
        echo "  nonorth=$n DIVERGED"
    else
        echo "  nonorth=$n capped"
    fi
    cd ..
done
