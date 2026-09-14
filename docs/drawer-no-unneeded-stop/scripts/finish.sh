#!/bin/bash
set -euo pipefail
cd /home/coder/share/real-robot-data-retime-close-on-retreat
export PYTHONPATH=$PWD/src:/home/coder/share/robo-visualize/src
export ROBOVISUALIZE_ASSETS=/home/coder/share/robo-visualize/src/robo_visualize/arms/piperx/assets
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/coder/share/real-robot-data-retime/.venv/bin/python
R=/home/coder/share/drawer-no-unneeded-stop-20260915
$PY "$R/render-all.py" > "$R/render.log" 2>&1
$PY -m real_robot_data_retime.uniform_drawer --config "$R/config.json" --phase finalize > "$R/finalize.log" 2>&1
$PY "$R/audit.py" > "$R/audit.log" 2>&1
$PY "$R/local-review-package/compress.py" > "$R/compress.log" 2>&1
$PY scripts/export_review_idle_masks.py --dataset "$R/dataset" --manifest "$R/local-review-package/review/manifest.json" --output "$R/local-review-package/review/idle-masks.json" --phase-boundaries "$R/local-review-package/review/close-preparation-boundaries.json" > "$R/masks.log" 2>&1
$PY "$R/package.py" > "$R/package.log" 2>&1
$PY "$R/upload-review.py" > "$R/upload-review.log" 2>&1
