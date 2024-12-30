# To vis the final evaluation results in a beautiful table

import os
import json
from prettytable import PrettyTable
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description="Visualization parameters")
    parser.add_argument('--root', '-r', type=str, default="./results/")
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--scene', '-s', nargs='*', help="If this list is empty, we will search for all scenes in the root dir")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--eval_path', '-e', type=str, default="eval/test")

    args = parser.parse_args()
    root_dir = os.path.join(args.root, args.dataset)

    if args.scene is not None:
        scenes = args.scene
    else:
        scenes = os.listdir(root_dir)

    info = []
    cnt, psnr_avg, ssim_avg, msssim_avg, lpivgg_avg, lpialex_avg = 0, 0., 0., 0., 0., 0.
    for scene in scenes:
        json_path = os.path.join(root_dir, scene, args.model_path, args.eval_path, "results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                cnt += 1
                data_ = json.load(f)
                # for key in data_.keys():
                temp_data = {}
                temp_data["name"] = os.path.join(scene)
                temp_data["psnr"] = round(data_["PSNR"], 2)
                psnr_avg += round(data_["PSNR"], 2)
                temp_data["ssim"] = round(data_["SSIM"], 4)
                ssim_avg += round(data_["SSIM"], 4)
                temp_data["msssim"] = round(data_["MS-SSIM"], 4)
                msssim_avg += round(data_["MS-SSIM"], 4)
                temp_data["lpivgg"] = round(data_["LPIPS-vgg"], 4)
                lpivgg_avg += round(data_["LPIPS-vgg"], 4)
                temp_data["lpialex"] = round(data_["LPIPS-alex"], 4)
                lpialex_avg += round(data_["LPIPS-alex"], 4)

                info.append(temp_data)

    info = sorted(info, key=lambda x: x["name"])
    info.append({"name": "Average", 
                 "psnr": round(psnr_avg / cnt, 2), 
                 "ssim": round(ssim_avg / cnt, 4), 
                 "msssim": round(msssim_avg / cnt, 4), 
                 "lpivgg": round(lpivgg_avg / cnt, 4), 
                 "lpialex": round(lpialex_avg / cnt, 4)})
    
    table = PrettyTable()
    table.field_names = ["Scene", "PSNR", "SSIM", "MS-SSIMS", "LPIPS-vgg", "LPIPS-alex"]

    for data in info:
        table.add_row([data["name"], data["psnr"], data["ssim"], data["msssim"], data["lpivgg"], data["lpialex"]])

    print("+++++++++++++++++++++++++++++++++++++")
    print(f"Visualizing {args.dataset} dataset: {args.model_path}")
    print("+++++++++++++++++++++++++++++++++++++")
    print(table)