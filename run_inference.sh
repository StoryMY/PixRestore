#!/usr/bin/env bash
python inference.py \
  -c models/pixrestore-s \
  -i input \
  -o outputs \
  --infer-steps 1 \
  --comparison-video outputs/comparison.mp4 \
  --dinov2-repository models/dinov2 \
  --dinov2-checkpoint models/dinov2_vits14_pretrain.pth \
  --cfg-scale 1.0 \
  --seed 42
