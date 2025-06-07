# Bỏ 2 dòng cuối nếu không push lên HuggingFace
python train.py \
  --task gen \
  --train_file train_augment_all_final_f1.csv \
  --valid_file /data/valid_multitask_gen.csv \
  --output_dir finet_3 \
  --init_model thevan2404/codet5-phase2 \
  --num_train_epochs 3 \
  --push_to_hub \
  --hub_model_id thevan2404/codet5-phase3