scene_name="standup"
max_keypoints=100
adaptive_points_num=100
time_freq=6
nearest_num=6
max_time=0.8
feature_amplify=0.5

model_path="./results/d-nerf_${max_time}/${scene_name}/finalVersion/"

CUDA_VISIBLE_DEVICES=0 python eval.py -m $model_path --max_points $max_keypoints --max_time $max_time \
    --adaptive_points_num $adaptive_points_num --ckpt_iteration 60000 --time_freq $time_freq \
    --nearest_num $nearest_num --feature_amplify $feature_amplify --norm_rotation 