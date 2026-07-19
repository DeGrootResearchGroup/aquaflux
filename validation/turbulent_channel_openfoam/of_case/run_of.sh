#!/bin/bash
# 2D resolved-wall kOmegaSST periodic channel, at a low and a high Re_tau. Runs inside the
# openfoam13 container and saves each converged field (+ cell centres) to ../runs/<name>/:
#   docker run --rm -v <study dir>:/work -w /work/of_case openfoam13:latest bash run_of.sh
# Ubar = 0.1335 (meanVelocityForce); Re_tau is set by nu. The wall-normal mesh (ny per half-block,
# grade = wall expansion ratio) is refined for the high-Re_tau point to keep the first cell y+ < 1.
set -e
cd "$(dirname "$0")"
RUNS=../runs
mkdir -p "$RUNS"

# Restore the tracked case files to their committed defaults on exit (the loop seds them per case),
# so `git status` stays clean after a run.
restore() {
    sed -i "s/^nu .*/nu  [0 2 -1 0 0 0 0] 2e-05;/" constant/physicalProperties
    sed -i "s/^ny  .*/ny  40;/" system/blockMeshDict
    sed -i "s/^grade     .*/grade     60;/" system/blockMeshDict
    sed -i "s/^gradeInv  .*/gradeInv  0.0166667;/" system/blockMeshDict
    sed -i "s/^endTime .*/endTime         3000;/" system/controlDict
}
trap restore EXIT

# name  nu        ny   grade   endTime
CASES=(
    "low   2e-5     40   60      3000"
    "high  1.57e-6  120  200     8000"
)

for c in "${CASES[@]}"; do
    read -r NAME NU NY GRADE ENDT <<< "$c"
    GINV=$(python3 -c "print(1.0/$GRADE)" 2>/dev/null || awk "BEGIN{print 1.0/$GRADE}")
    sed -i "s/^nu .*/nu  [0 2 -1 0 0 0 0] $NU;/" constant/physicalProperties
    sed -i "s/^ny  .*/ny  $NY;/" system/blockMeshDict
    sed -i "s/^grade     .*/grade     $GRADE;/" system/blockMeshDict
    sed -i "s/^gradeInv  .*/gradeInv  $GINV;/" system/blockMeshDict
    sed -i "s/^endTime .*/endTime         $ENDT;/" system/controlDict

    rm -rf 0 [1-9]* processor* postProcessing log.* 2>/dev/null || true
    cp -r 0.orig 0
    blockMesh > log.blockMesh 2>&1
    foamRun > log.foamRun 2>&1 || true
    foamPostProcess -func writeCellCentres -latestTime > log.cc 2>&1 || true

    T=$(foamListTimes -latestTime)
    dst="$RUNS/$NAME"
    rm -rf "$dst"; mkdir -p "$dst"
    cp "$T"/{Ccx,Ccy,U,k,omega,nut} "$dst"/
    echo "$NU" > "$dst/nu"
    echo "$NAME: nu=$NU ny=$NY Re_tau-mesh=$T  $(grep -o 'SIMPLE solution converged in [0-9]* iterations' log.foamRun || echo 'ran to endTime')" | tee -a "$RUNS/summary.txt"
done
