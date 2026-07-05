#!/usr/bin/env bash
set -euo pipefail

# Run from the StructEdit project root:
#   bash run_edit_verification.sh
#
# This produces:
#   1) edits from every successful baseline bbox
#   2) VG official bbox images
#   3) edits from VG official bbox + SAM2
#   4) final side-by-side comparison grids

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"

BASELINE_RESULTS="${BASELINE_RESULTS:-outputs/selection_baselines_all/baseline_results.jsonl}"
TASKS_JSONL="${TASKS_JSONL:-scripts/edit_verification_tasks.jsonl}"

IMAGE_ROOT="${IMAGE_ROOT:-data/vg/no_edit}"
IMAGE_EXT="${IMAGE_EXT:-jpg}"
SCENE_GRAPH_ROOT="${SCENE_GRAPH_ROOT:-data/vg/scene_graphs_vg}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/edit_verification_baseline_vs_vg}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-scripts/run_pipeline_Powerpaint_external_mask.py}"

# Official SAM2 usage normally uses the config name below, while the repository
# itself is added to PYTHONPATH.
SAM2_REPO="${SAM2_REPO:-checkpoints/sam2}"
SAM2_CONFIG="${SAM2_CONFIG:-sam2_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-checkpoints/sam2_weights/sam2_hiera_large.pt}"

export PYTHONPATH="$(pwd)/${SAM2_REPO}:${PYTHONPATH:-}"

"${PYTHON}" scripts/run_edit_verification_from_baseline_and_vg.py \
  --baseline-results "${BASELINE_RESULTS}" \
  --tasks-jsonl "${TASKS_JSONL}" \
  --image-root "${IMAGE_ROOT}" \
  --image-ext "${IMAGE_EXT}" \
  --scene-graph-root "${SCENE_GRAPH_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --sam2-config "${SAM2_CONFIG}" \
  --sam2-checkpoint "${SAM2_CHECKPOINT}" \
  --device "${DEVICE}" \
  --python "${PYTHON}" \
  --pipeline-script "${PIPELINE_SCRIPT}" \
  --run-powerpaint \
  --columns 3

echo
echo "Done."
echo "Baseline edits:       ${OUTPUT_ROOT}/baseline_edits"
echo "VG official boxes:    ${OUTPUT_ROOT}/vg_official_boxes"
echo "VG official edits:    ${OUTPUT_ROOT}/vg_official_edits"
echo "Comparison grids:     ${OUTPUT_ROOT}/comparisons"
