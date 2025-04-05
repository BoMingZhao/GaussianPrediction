root="./results"
model_path="finalVersion"

### Max time == 1 (reconsturction results)
max_time=0.8
dataset="d-nerf_${max_time}"
python show.py -r $root -d $dataset -m $model_path

### Max time < 1 (prediction results)
# max_time=0.8
# dataset="d-nerf_${max_time}"
# eval_path="input10_stage6_noise0_epoch2001/[metrics]Predicted_by_GCN_on_test_views"
# python show.py -r $root -d $dataset -m $model_path --eval_path $eval_path