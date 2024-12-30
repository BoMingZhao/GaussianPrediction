scene_name="cut-lemon-new"
time_freq=10
max_keypoints=100
adaptive_points_num=200
max_time=0.8
noise_init=0.5
R_ratio=1.0

num_stage=16
input_size=10
noise_step=500
epoch=1501
linear_size=128
exp_name="input${input_size}_stage${num_stage}_hidden${linear_size}_noise${noise_init}_RNoise${R_ratio}_noiseStep${noise_step}_epoch${epoch}"


model_path="./results/HyperNeRF_${max_time}/${scene_name}/finalVersion/"

CUDA_VISIBLE_DEVICES=0 python train_GCN.py -m $model_path --max_points $max_keypoints --time_freq $time_freq \
    --adaptive_points_num $adaptive_points_num --ckpt_iteration 70000 --max_time $max_time \
    --num_stage $num_stage --noise_step $noise_step --noise_init $noise_init --linear_size $linear_size --input_size $input_size \
    --Rscale_ratio $R_ratio --batch_size 32 --epoch $epoch --exp_name $exp_name --predict_more --metrics --norm_rotation --step_opacity --cam_id=-16