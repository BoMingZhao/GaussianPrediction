#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import  numpy as np
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim

def readImages(renders_dir, gt_dir, resize=False, resize_ratio=0.5):
    renders = []
    gts = []
    image_names = []
    image_gt_names = []

    render_fname_list = sorted(os.listdir(renders_dir))
    gt_fname_list = sorted(os.listdir(gt_dir))

    for fname in render_fname_list:
        if "depth" in fname:
            continue
        render = Image.open(os.path.join(renders_dir, fname))
        size = render.size
        W = int(size[0] * resize_ratio)
        H = int(size[1] * resize_ratio)
        new_size = (W, H)
        if resize:
            render = render.resize(new_size)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    for fname in gt_fname_list:
        gt = Image.open(os.path.join(gt_dir, fname))
        size = gt.size
        W = int(size[0] * resize_ratio)
        H = int(size[1] * resize_ratio)
        new_size = (W, H)
        if resize:
            gt = gt.resize(new_size)
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_gt_names.append(fname)
    return renders, gts, image_names

def evaluate_twodir(gt_dir, renders_dir):    
    try:
        full_dict = {}
        per_view_dict = {}
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []
        lpipsa = []
        ms_ssims = []
        Dssims = []
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
            ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
            lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
            Dssims.append((1-ms_ssims[-1])/2)

        print("Scene: ", renders_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("Scene: ", renders_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("Scene: ", renders_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("Scene: ", renders_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
        print("Scene: ", renders_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
        print("Scene: ", renders_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

        full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                "D-SSIM": torch.tensor(Dssims).mean().item()},

                                            )
        per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                    "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                    "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                    }
                                                )

        with open(renders_dir + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(renders_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
        
    except Exception as e:
        print("Unable to compute metrics for model", renders_dir)
        raise e

def evaluate(model_paths, resize=False, resize_ratio=0.5):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for method in os.listdir(model_paths):
        if not os.path.isdir(os.path.join(model_paths, method)):
            continue
        try:
            print("Method:", method)

            full_dict[method] = {}
            per_view_dict[method] = {}
            full_dict_polytopeonly[method] = {}
            per_view_dict_polytopeonly[method] = {}

            method_dir = os.path.join(model_paths, method)
            gt_dir = os.path.join(method_dir, "gt") 
            renders_dir = os.path.join(method_dir, "renders") 
            os.makedirs(os.path.join(method_dir, "deltas"), exist_ok=True)
            renders, gts, image_names = readImages(renders_dir, gt_dir, resize=resize, resize_ratio=resize_ratio)

            ssims, psnrs, lpipss, lpipsa, ms_ssims, Dssims = [], [], [], [], [], []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                Dssims.append((1-ms_ssims[-1])/2)
                error = (renders[idx] - gts[idx])[0].permute(1, 2, 0).abs().cpu().numpy()
                error_img = Image.fromarray((error * 255).astype(np.uint8))
                error_img.save(os.path.join(method_dir, "deltas", "{0:05d}.jpg".format(idx)))

            print("Scene: ", model_paths,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("Scene: ", model_paths,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("Scene: ", model_paths,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("Scene: ", model_paths,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
            print("Scene: ", model_paths,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
            print("Scene: ", model_paths,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

            full_dict[method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                    "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                    "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                    "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                    "D-SSIM": torch.tensor(Dssims).mean().item()})
            
            per_view_dict[method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                        "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                        "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                        "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                        "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)}})

            with open(model_paths + "/results.json", 'w') as fp:
                json.dump(full_dict[method], fp, indent=True)
            with open(model_paths + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[method], fp, indent=True)
                
        except Exception as e:
            print("Unable to compute metrics for model", model_paths)
            raise e

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gt_paths', '-g', type=str, default="")
    parser.add_argument('--split', action='store_true', default=False)
    parser.add_argument('--resize', action='store_true', default=False)
    args = parser.parse_args()
    if args.split:
        evaluate_twodir(args.gt_paths, args.model_paths[0])
    else:
        evaluate(args.model_paths)