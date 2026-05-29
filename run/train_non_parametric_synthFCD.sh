#!/bin/bash
set -euo pipefail

: "${RUN_NAME:?RUN_NAME is not set}"
: "${TRAINING_TIME_MINUTES:?TRAINING_TIME_MINUTES is not set}"
: "${EXPERIMENTS_DIR:?EXPERIMENTS_DIR is not set}"

if [ -n "${CKPT_PATH:-}" ]; then
  ckpt_arg=(--ckpt_path "$CKPT_PATH")
else
  ckpt_arg=()
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
L2S_RUN_NAME="$RUN_NAME" \
L2S_TIME_LIMIT_MINUTES="$TRAINING_TIME_MINUTES" \
python ../scripts/train_non_parametric_synthFCD.py fit \
  "${ckpt_arg[@]}" \
  \
  --data.batch_size 2 \
  --data.num_workers 4 \
  --data.eval 0.2 \
  --data.fcd_intensity_range "[0.02, 0.3602]" \
  --data.fcd_tail_range "[14, 29]" \
  --data.split_seed 42 \
  --data.use_extra_data true \
  \
  --model.native_synthesis false \
  --model.flair_modality true \
  --model.seg_nb_levels 6 \
  --model.seg_features "[16,32,64,128,256,512]" \
  --model.time_limit_minutes "$TRAINING_TIME_MINUTES" \
  --model.val_diagnostics_interval 10 \
  --model.debug_subject_ids '["sub-00001", "sub-00033", "sub-00044", "sub-00002", "sub-00058", "sub-00065"]' \
  \
  --trainer.default_root_dir "$EXPERIMENTS_DIR" \
  --trainer.max_epochs 2000 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --trainer.enable_progress_bar false \
  --trainer.log_every_n_steps 5 \
  --trainer.num_sanity_val_steps 0 \
  \
  --checkpoint.save_top_k 1 \
  --checkpoint.monitor eval_loss \
  --checkpoint.mode min \
  --checkpoint.save_last true \
  \
  --seed_everything 0