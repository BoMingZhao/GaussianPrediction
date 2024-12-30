scene_name="chickchicken"
time_freq=8
max_keypoints=100
adaptive_points_num=100
nearest_num=6
max_time=1.0
feature_amplify=5
model_path="./results/HyperNeRF_${max_time}/${scene_name}/finalVersion"

CUDA_VISIBLE_DEVICES=0 python eval.py -m $model_path --max_points $max_keypoints --time_freq $time_freq \
    --adaptive_points_num $adaptive_points_num --ckpt_iteration 70000 --feature_amplify $feature_amplify \
    --nearest_num $nearest_num --max_time $max_time --norm_rotation