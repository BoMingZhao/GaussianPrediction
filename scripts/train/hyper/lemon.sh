scene_name="cut-lemon-new"
time_freq=10
feature_amplify=5
max_keypoints=100
adaptive_points_num=200
nearest_num=6
max_time=1.0
source_path="./datasets/HyperNeRF/${scene_name}/"
model_path="./results/HyperNeRF_${max_time}/${scene_name}/finalVersion"
position_lr_max_steps=40000
data_device="cpu"
step_opacity_iteration=5000

# Train
CUDA_VISIBLE_DEVICES=0 python train.py -s $source_path \
    -m $model_path --max_points $max_keypoints --adaptive_points_num $adaptive_points_num  \
    --iterations 70000 --test_iterations 70000 --jointly_iteration 1000  --time_freq $time_freq\
    --densify_from_iter 5000 --save_iterations 29998 70000 --checkpoint_iterations 29998 70000 \
    --densify_until_iter 15000  --opacity_reset_interval 3000000 --max_time $max_time --norm_rotation --step_opacity \
    --position_lr_max_steps $position_lr_max_steps --data_device $data_device --step_opacity_iteration $step_opacity_iteration \
    --eval --use_time_decay --nearest_num $nearest_num --adaptive_interval 1000 --feature_amplify $feature_amplify 