# render deformable image from different cam or single view

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

import torch
import numpy as np
from scene import Scene
import os
import time
import cv2
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer import render_motion
from scene.deformable_field import positional_encoding
import torchvision
from metrics import evaluate
from utils.camera_utils import transpose, pi, interpolation_pose
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.visualizer_utils import *
from gaussian_renderer import GaussianModel
import copy
from scene.cameras import Camera
from utils.prepare.makeVideo import make_video
from options.gaussian_option import Gaussian_Options
from tqdm import tqdm

def project_trajectory(gaussians, view, tidx, time_freq, iteration, steps=60):
    H, W = tidx.shape[0], tidx.shape[1]
    image = np.zeros([H, W, 3], dtype=np.float32)
    times = torch.linspace(0, view.time.item(), steps)
    tidx = tidx.view([-1])
    mask = tidx != -1
    tidx_valid = tidx[mask]
    xyz = gaussians.get_xyz[tidx_valid]
    color = np.random.rand(xyz.shape[0], 3)

    R = torch.from_numpy(view.R.T).to(torch.float32).cuda()
    t = torch.from_numpy(view.T).to(torch.float32).cuda()
    focal = fov2focal(view.FoVx, W)
    K = torch.ones([3, 3], dtype=torch.float32).cuda()
    K[0, 0] = focal
    K[0, 2] = W / 2
    K[1, 1] = focal
    K[1, 2] = H / 2
    K[2, 2] = 1.0
    for i in range(len(times)):
        time = (positional_encoding(times[i:i+1].cuda(), time_freq)).unsqueeze(0)
        gaussians.forward(time, iteration)
        xyz_time = gaussians.all_xyz_motion[tidx_valid] + xyz
        xyz_cam, _ = pi(K, transpose(R, t, xyz_time))
        xyz_cam = xyz_cam.cpu().numpy().astype(np.int32)
        if i != 0:
            for j in range(xyz_cam.shape[0]):
                point1_x = xyz_cam[j][0]
                point1_y = xyz_cam[j][1]
                point1 = (point1_x, point1_y)
                point2_x = xyz_previes[j][0]
                point2_y = xyz_previes[j][1]
                point2 = (point2_x, point2_y)
                cv2.line(image, point1, point2, color=color[j])
        xyz_previes = xyz_cam.copy()
    return image

def render_video(model_path, name, iteration, views, gaussians, pipeline, background, opt, args, interpolation=5):
    render_path = os.path.join(model_path + "eval", name, "ours_{}".format(iteration), "renders_video")
    makedirs(render_path, exist_ok=True)

    time_avg = 0.
    step = 2 if "vrig" in model_path else 1
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        previous_view = view
        if idx != 0:
            if idx % step != 0:
                continue
            previous_view = views[idx - step]
        if isinstance(view, dict): # Use optical flow
            view = view["cam"]
            previous_view = previous_view["cam"]

        rendering_time = torch.from_numpy(view.time).to(torch.float32).cuda()
        previous_rendering_time = torch.from_numpy(previous_view.time).to(torch.float32).cuda()

        time_slice = (rendering_time - previous_rendering_time) / interpolation
        for frame in range(1, interpolation + 1):
            new_view = copy.deepcopy(previous_view)
            id = 0 if idx == 0 else (frame + ((idx // step) - 1) * interpolation)
            time_inter = previous_rendering_time + frame * time_slice
            new_t, new_R = interpolation_pose(view=view, previous_view=previous_view, interpolation_ratio=frame / interpolation)
            new_view.R = new_R
            new_view.T = new_t
            new_view = Camera.from_Camera(new_view)
            torch.cuda.synchronize()
            start_time = time.time()
            render_pkg = render(new_view, gaussians, pipeline, background, delta=None, time=time_inter, it=iteration)
            torch.cuda.synchronize()
            end_time = time.time()
            time_avg += end_time - start_time
            rendering, depth, tidx = render_pkg["render"], render_pkg["depth"], render_pkg["tidx"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
            if idx == 0:
                break
    output_name = os.path.basename(os.path.dirname(model_path[:-1])) + ".mp4"
    img_num = interpolation * ((len(views) - 1) // step) + 1
    make_video(render_path, image_num=img_num, fps=120, concat_delta=False, output_name=output_name)
    time_avg /= img_num
    fps_avg = 1. / time_avg
    print(f"Rendering AVG Times: {time_avg:.5f}, FPS: {fps_avg:.5f}")

def render_kpts(model_path, exp_name, name, iteration, views, gaussians, pipeline, background, kpts, kpts_rotation, 
                metrics=False, view_id=None):
    eval_path = os.path.join(model_path, exp_name, name)
    render_path = os.path.join(eval_path, "ours_{}".format(iteration), "renders")
    
    time_ = torch.from_numpy(views[0].time).to(torch.float32).cuda()
    _, _, _, _, weights_xyz, weights_r = gaussians(time_, iteration, return_weights=True)
    life_opacity = gaussians.lifecycle_opacity

    if view_id is not None:
        render_path = os.path.join(render_path, f"view{view_id}")
    makedirs(render_path, exist_ok=True)

    if metrics:
        gts_path = os.path.join(eval_path, "ours_{}".format(iteration), "gt")
        makedirs(gts_path, exist_ok=True)

    for i in tqdm(range(len(kpts))):
        gaussian_xyz, gaussian_kpts = gaussians.get_xyz, gaussians.super_gaussians

        xyz_final = gaussian_xyz + (weights_xyz @ (kpts[i] - gaussian_kpts))
        delta_r = gaussians.get_rotation_(gaussians.rotation_activation(weights_r @ kpts_rotation[i]))

        if metrics:
            view = views[i]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(i) + ".png"))
        elif view_id is not None:
            view = views[view_id]
        else:
            position = 2 if ((id // (len(views) // 2)) % 2 == 0) else -2
            view = views[(id % (len(views) // 2)) * position]

        render_pkg = render_motion(view, gaussians, pipeline, background, xyz_t=xyz_final, r_t=delta_r, opacity=life_opacity)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))

    return eval_path, render_path

def render_trainSequence(model_path, name, iteration, train_views, gaussians, pipeline, background, opt, args, test_views, freeze_view_number=5):
    render_path = os.path.join(model_path + "eval", name, "ours_{}".format(iteration), "renders")
    render_path = os.path.join(render_path, f"view_{freeze_view_number:03d}")
    gts_path = os.path.join(model_path + "eval", name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    time_avg = 0.
    view_freeze = test_views[freeze_view_number]
    for idx, view in enumerate(tqdm(train_views, desc="Rendering progress")):
        if isinstance(view, dict): # Use optical flow
            view = view["cam"]
        time_ = torch.from_numpy(view.time).to(torch.float32).cuda()
        
        start_time = time.time()
        torch.cuda.synchronize()
        render_pkg = render(view_freeze, gaussians, pipeline, background, delta=None, time=time_, it=iteration)
        torch.cuda.synchronize()
        end_time = time.time()
        time_avg += end_time - start_time

        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


    time_avg /= len(train_views)
    fps_avg = 1. / time_avg
    print(f"Rendering AVG Times: {time_avg:.5f}, FPS: {fps_avg:.5f}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, opt, args):
    eval_path = os.path.join(model_path, "eval", name)
    render_path = os.path.join(eval_path, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(eval_path, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    np.savetxt(os.path.join(model_path, "kpts_fps.txt"), gaussians.get_superGaussians[:args.max_points].detach().cpu().numpy())
    if args.adaptive_points_num > 0:
        np.savetxt(os.path.join(model_path, "kpts_incre.txt"), gaussians.get_superGaussians[args.max_points:].detach().cpu().numpy())

    time_avg = 0.
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if isinstance(view, dict): # Use optical flow
            view = view["cam"]
        time_ = torch.from_numpy(view.time).to(torch.float32).cuda()

        start_time = time.time()
        torch.cuda.synchronize()
        render_pkg = render(view, gaussians, pipeline, background, delta=None, time=time_, it=iteration)
        torch.cuda.synchronize()
        end_time = time.time()
        time_avg += end_time - start_time
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    time_avg /= len(views)
    fps_avg = 1. / time_avg
    print(f"Rendering AVG Times: {time_avg:.5f}, FPS: {fps_avg:.5f}")

    return eval_path

def render_sets(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, ckpt_iteration : int, args):
    with torch.no_grad():
        xyz_freq = opt.xyz_freq
        time_freq = opt.time_freq

        time_input_dim = 2 * time_freq
        xyz_input_dim = 6 * xyz_freq

        gaussians = GaussianModel(dataset.sh_degree, args)
        gaussians.set_inputDim(time_input_dim, xyz_input_dim)
        (model_params, opt_dict, first_iter) = torch.load(os.path.join(dataset.model_path, f"chkpnt{ckpt_iteration:d}.pth"))
        gaussians.N_pcd_init = model_params["_xyz"].shape[0]
        gaussians.active_sh_degree = gaussians.max_sh_degree
        gaussians.final_kpts_num = model_params["super_gaussians"].shape[0] if "super_gaussians" in model_params.keys() else None
        
        scene = Scene(dataset, gaussians, shuffle=False, ratio=args.ratio)
        gaussians.restore(opt_dict, opt, first_iter)
        gaussians.load_state_dict(model_params, strict=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.render_train:
            for view_num in args.train_view:
                render_trainSequence(dataset.model_path, "train", first_iter, scene.getTrainCameras(), gaussians, 
                                pipeline, background, opt, args, scene.getTestCameras(), freeze_view_number=view_num)
        elif args.render_video:
            render_video(dataset.model_path, "video", first_iter, scene.getRenderCameras(), gaussians, 
                        pipeline, background, opt, args, interpolation=args.interpolation)
        else:
            eval_path = render_set(dataset.model_path, "test", first_iter, scene.getTestCameras(), gaussians, pipeline, background, opt, args)
            evaluate(eval_path, resize_ratio=args.resize)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    op = OptimizationParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    options.eval()
    parser = options.get_parser()

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(model.extract(args), op.extract(args), args.ckpt_iteration, pipeline.extract(args), args.ckpt_iteration, args)
