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

./scripts/train/hyper/lemon.sh 
wait
./scripts/predict/hyper/lemon.sh 
wait

./scripts/train/hyper/chickchicken.sh 
wait
./scripts/predict/hyper/chickchicken.sh 
wait

./scripts/train/hyper/printer.sh 
wait
./scripts/predict/hyper/printer.sh 
wait

./scripts/train/hyper/torchocolate.sh 
wait
./scripts/predict/hyper/torchocolate.sh 
wait

echo "Done"