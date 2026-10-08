CONFIG=configs/finetune.yaml NUM_PROCESSES=1 bash scripts/train.sh \
  --manifest /path/to/train.jsonl \
  --pretrained models/pixrestore-s/clean_weights/ema_model.safetensors \
  --dinov2-repository models/dinov2 \
  --dinov2-checkpoint models/dinov2_vits14_pretrain.pth \
  --output-dir outputs/finetune \
  --test-manifest /path/to/val.jsonl \
  > train.log 2>&1
