python main.py \
  --image_dir R2GenCMN/data/iu_xray/images \
  --ann_path data/iu_xray/annotation.json \
  --dataset_name iu_xray \
  --max_seq_length 40 \
  --threshold 3 \
  --batch_size 4 \
  --epochs 10 \
  --save_dir results/iu_xray_test \
  --step_size 50 \
  --gamma 0.1 \
  --seed 9223 \
  --log_period 50 \
  --problem_dim 1357 \
  
