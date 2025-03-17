echo "Start training and evaluation process"
echo "Please make sure your scripts [max_time] is 1.0 "

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
./scripts/eval/hyper/lemon.sh 
wait

./scripts/train/hyper/chickchicken.sh 
wait
./scripts/eval/hyper/chickchicken.sh 
wait

./scripts/train/hyper/printer.sh 
wait
./scripts/eval/hyper/printer.sh 
wait

./scripts/train/hyper/torchocolate.sh 
wait
./scripts/eval/hyper/torchocolate.sh 
wait

./scripts/utils/hyper_show.sh