# Bỏ 2 dòng cuối nếu không push lên HuggingFace
python train.py \
  --task denoise \
  --train_file /data/train_multitask_complete.csv \
  --valid_file /data/valid_multitask_complete.csv \
  --output_dir finet_1 \
  --init_model Salesforce/codet5-base \
  --num_train_epochs 3 \
  --push_to_hub \
  --hub_model_id thevan2404/codet5-phase1