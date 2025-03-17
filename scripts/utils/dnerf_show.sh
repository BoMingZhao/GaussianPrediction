root="./results"
model_path="finalVersion"
dataset="d-nerf_0.8"
eval_path="input10_stage6_noise0_epoch2001/[metrics]Predicted_by_GCN_on_test_views"

python show.py -r $root -d $dataset -m $model_path --eval_path $eval_path