#!/usr/bin/env bash
#
# The resolution studies, one case at a time through validation/run_case.sh: the background grid at
# the baseline tessellation, and each tessellation setting on the 8 mm grid with the rest at the
# baseline (lamp rings 8 mm, 24 sectors, wall edges 30 mm, 8 sides per hole). Every run writes its
# fields, data and figures to work/runs/<tag>; compare.py reads them all.
#
# Usage: validation/tubular_reactor/study.sh [first-step]   (steps are numbered in the log)
set -u
cd "$(dirname "$0")/../.."
RUN=validation/tubular_reactor/run.py
first=${1:-1}
step=0

case_() {  # case_ <label> <env assignments...>
  step=$((step + 1))
  local label=$1; shift
  if [ "$step" -lt "$first" ]; then echo "[study] step $step ($label) skipped"; return; fi
  validation/run_case.sh --wait >/dev/null 2>&1  # whatever is already running finishes first
  echo "[study] step $step: $label ($*) $(date '+%H:%M:%S')"
  env "$@" validation/run_case.sh "$RUN" --wait >/dev/null 2>&1
  echo "[study] step $step: $label exited $? $(date '+%H:%M:%S')"
}

# The runs made before the study wrote figures: move them under their tags and redraw theirs.
validation/run_case.sh --wait >/dev/null 2>&1
W=validation/tubular_reactor/work
mkdir -p "$W/runs"
for h in 8 4; do
  if [ -d "$W/h${h}mm" ] && [ ! -d "$W/runs/h${h}_ax8_s24_w30_t8" ]; then
    mv "$W/h${h}mm" "$W/runs/h${h}_ax8_s24_w30_t8"
  fi
done
case_ "figures h8" TUBULAR_H=0.008 TUBULAR_FIGURES_ONLY=1
case_ "figures h4" TUBULAR_H=0.004 TUBULAR_FIGURES_ONLY=1
# The plate's shadow resolved finely over the middle of the plate.
step=$((step + 1))
if [ "$step" -ge "$first" ]; then
  validation/run_case.sh --wait >/dev/null 2>&1
  echo "[study] step $step: plate close-up 0.25 mm $(date '+%H:%M:%S')"
  TUBULAR_PLANE_SPACING=0.00025 TUBULAR_PLANE_HALF=0.04 \
    validation/run_case.sh validation/tubular_reactor/plate_shadow.py --wait >/dev/null 2>&1
  echo "[study] step $step: plate close-up exited $? $(date '+%H:%M:%S')"
fi
# Lamp rings along the axis.
case_ "lamp rings 16 mm" TUBULAR_H=0.008 TUBULAR_AXIAL=0.016
case_ "lamp rings 4 mm" TUBULAR_H=0.008 TUBULAR_AXIAL=0.004
case_ "lamp rings 2 mm" TUBULAR_H=0.008 TUBULAR_AXIAL=0.002
# Lamp sectors round the sleeve.
case_ "lamp sectors 48" TUBULAR_H=0.008 TUBULAR_SECTORS=48
# Wall triangles.
case_ "walls 60 mm" TUBULAR_H=0.008 TUBULAR_WALL=0.06
case_ "walls 20 mm" TUBULAR_H=0.008 TUBULAR_WALL=0.02
case_ "holes 12 sides" TUBULAR_H=0.008 TUBULAR_PER_TURN=12
# The background grid.
case_ "grid 6 mm" TUBULAR_H=0.006
case_ "grid 3 mm" TUBULAR_H=0.003
echo "[study] done $(date '+%H:%M:%S')"
