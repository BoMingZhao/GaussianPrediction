scene_name="hook"
max_keypoints=100
time_freq=6
adaptive_points_num=100
nearest_num=6
feature_amplify=0.5

# gcn param
num_stage=6
noise_init=0
noise_step=100
input_size=10
epoch=2001
exp_name="input${input_size}_stage${num_stage}_noise${noise_init}_epoch${epoch}"

model_path="./results/d-nerf_${max_time}/${scene_name}/finalVersion/"

CUDA_VISIBLE_DEVICES=0 python train_GCN.py  -m $model_path --max_points $max_keypoints \
    --adaptive_points_num $adaptive_points_num --ckpt_iteration 60000 --time_freq $time_freq \
    --max_time $max_time --nearest_num $nearest_num  --feature_amplify $feature_amplify \
    --num_stage $num_stage --noise_step $noise_step --noise_init $noise_init  --input_size $input_size \
    --epoch $epoch --exp_name $exp_name --predict_more --metrics --norm_rotation