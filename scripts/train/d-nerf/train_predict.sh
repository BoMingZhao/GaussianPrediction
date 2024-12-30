echo "Start training and prediction process"
echo "Please make sure your scripts [max_time] is 0.8"

read -p "Do you want to continue? (yes/no): " user_input

# 判断输入
if [ "$user_input" = "yes" ]; then
    echo "Continuing execution..."
    # 在这里继续执行你的脚本逻辑
else
    echo "Exiting..."
    exit 1
fi

# Train
./scripts/train/d-nerf/bouncingballs.sh
wait
./scripts/predict/d-nerf/bouncingballs.sh
wait

./scripts/train/d-nerf/hellwarrior.sh 
wait
./scripts/predict/d-nerf/hellwarrior.sh 
wait

./scripts/train/d-nerf/hook.sh 
wait
./scripts/predict/d-nerf/hook.sh 
wait

./scripts/train/d-nerf/mutant.sh 
wait
./scripts/predict/d-nerf/mutant.sh 
wait

./scripts/train/d-nerf/standup.sh 
wait
./scripts/predict/d-nerf/standup.sh 
wait

./scripts/train/d-nerf/jumping.sh 
wait
./scripts/predict/d-nerf/jumping.sh 
wait

./scripts/train/d-nerf/trex.sh
wait
./scripts/predict/d-nerf/trex.sh
wait

