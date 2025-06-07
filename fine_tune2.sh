# Bỏ 2 dòng cuối nếu không push lên HuggingFace
python train.py \
  --task gen \
  --train_file /data/train_multitask_gen.csv \
  --valid_file /data/valid_multitask_gen.csv \
  --output_dir finet_2 \
  --init_model thevan2404/codet5-phase1 \
  --num_train_epochs 3 \
  --push_to_hub \
  --hub_model_id thevan2404/codet5-phase2