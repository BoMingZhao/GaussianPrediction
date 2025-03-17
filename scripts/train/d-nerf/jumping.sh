if [ $# -eq 0 ]; then
    echo "Seed: 1*2024"
    seed=1
else
    seed=$1 
fi

scene_name="jumpingjacks"
max_keypoints=100
adaptive_points_num=100
time_freq=6
nearest_num=6
source_path="./datasets/d-nerf/data/${scene_name}/"
max_time=1.0
model_path="./results/d-nerf_${max_time}/${scene_name}/finalVersion/"
position_lr_max_steps=40000
adaptive_from_iter=3000
feature_amplify=0.5

# Train
CUDA_VISIBLE_DEVICES=0 python train.py -s $source_path \
    -m $model_path --max_points $max_keypoints --adaptive_points_num $adaptive_points_num \
    --iterations 60000 --test_iterations 60000 --jointly_iteration 1000 --time_freq $time_freq \
    --densify_from_iter 3000 --densify_until_iter 20000 --norm_rotation \
    --save_iterations 29999 60000 --checkpoint_iterations 29999 60000 --max_time $max_time \
    --position_lr_max_steps $position_lr_max_steps --adaptive_from_iter $adaptive_from_iter --seed $seed \
    --eval --nearest_num $nearest_num --adaptive_interval 500 --feature_amplify $feature_amplify