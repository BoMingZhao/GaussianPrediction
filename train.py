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

import os
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
import torch
import torchvision
from options.gaussian_option import Gaussian_Options
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene.deformable_field import positional_encoding
import sys
from scene import Scene, GaussianModel
import uuid
from shutil import copyfile
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, args, batch):
    first_iter = 0

    xyz_freq = opt.xyz_freq
    time_freq = opt.time_freq

    time_input_dim = 2 * time_freq
    xyz_input_dim = 6 * xyz_freq

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, args)
    gaussians.set_inputDim(time_input_dim, xyz_input_dim)
    if args.start_checkpoint:
        (model_params, opt_dict, first_iter) = torch.load(args.start_checkpoint)
        gaussians.N_pcd_init = model_params["_xyz"].shape[0]
        gaussians.active_sh_degree = gaussians.max_sh_degree
        gaussians.final_kpts_num = model_params["super_gaussians"].shape[0] if "super_gaussians" in model_params.keys() else None
    scene = Scene(dataset, gaussians, ratio=args.ratio) # This ratio only used in HyperNeRF dataset to select your training img's resolution
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        gaussians.restore(opt_dict, opt, first_iter)
        gaussians.load_state_dict(model_params, strict=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    batch_loss = []
    batch_viewspace_point_tensor = []
    batch_radii = []
    batch_visibility_filter = []

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        max_frame = scene.total_frame

        decay_noise = torch.randn([1], device="cuda") * args.time_noise_ratio / max_frame * (1 - min(1, iteration / args.time_noise_iteration))
        if args.use_time_decay:
            if iteration >= gaussians.second_stage_iter:
                decay_noise = torch.randn([1], device="cuda") * args.time_noise_ratio / max_frame * (1 - min(1, (iteration - gaussians.second_stage_iter) / (args.time_noise_iteration * 2)))
        else:
            decay_noise = torch.zeros_like(decay_noise)

        time_ = torch.from_numpy(viewpoint_cam.time).to(torch.float32).to("cuda") + decay_noise

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, delta=None, time=time_, it=iteration) # image shape
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        psnr_ = psnr(image, gt_image).mean().double()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss += gaussians.get_loss(iteration)

        # Batch operation
        batch_loss += [loss]
        batch_radii += [radii.unsqueeze(0)]
        batch_visibility_filter += [visibility_filter.unsqueeze(0)]
        batch_viewspace_point_tensor += [viewspace_point_tensor]

        if len(batch_loss) == batch:
            loss_ = torch.stack(batch_loss, dim=0).sum()
            loss_.backward()

            radii = torch.cat(batch_radii,0).max(dim=0).values
            visibility_filter = torch.cat(batch_visibility_filter).any(dim=0)
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(batch_viewspace_point_tensor)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + batch_viewspace_point_tensor[idx].grad

            batch_loss.clear()
            batch_radii.clear()
            batch_visibility_filter.clear()
            batch_viewspace_point_tensor.clear()
        else:
            continue

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{ema_psnr_for_log:.{3}f}",
                                          "p_num": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), [], scene, render, (pipe, background))
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if iteration % 5000 == 0: # save training images
                save_path = os.path.join(args.model_path, "train_imgs")
                print(f"saving to {save_path}")
                os.makedirs(save_path, exist_ok=True)
                torchvision.utils.save_image(image, os.path.join(save_path, 'render_{0:05d}'.format(iteration) + ".png"))
                gt = gt_image[0:3, :, :]
                torchvision.utils.save_image(gt, os.path.join(save_path, 'gt_{0:05d}'.format(iteration) + ".png"))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # In our scenes, max_gaussian_size equal to 200k is a good trade off between rendering quality and efficiency
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < args.max_gaussian_size:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            if iteration < args.adaptive_end_iter + gaussians.second_stage_iter and gaussians.super_gaussians.shape[0] < args.max_points + args.adaptive_points_num:
                # We use the first stage teach 
                if gaussians.second_stage:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > args.adaptive_from_iter + gaussians.second_stage_iter and iteration % args.adaptive_interval == 0:
                    if gaussians.new_xyz is not None: # which means teach model is activate
                        gaussians.densification_motion_postfix(gaussians.new_xyz, gaussians.new_motion_feature)
                        gaussians.new_kpts_init()
                    if args.densify_from_grad == "True":
                        gaussians.densify_kpts(opt.densify_grad_threshold, mode="down_sampling")
                    print(f"At iteration {iteration}, there are {gaussians.super_gaussians.shape[0]} super Gaussians!")
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.state_dict(), gaussians.optimizer.state_dict(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    file_up_path = os.path.join(args.model_path, 'recording')
    os.makedirs(file_up_path, exist_ok=True)
    cur_dir = os.path.join(file_up_path, "scene")
    os.makedirs(cur_dir, exist_ok=True)
    copyfile("train.py", os.path.join(file_up_path, "train.py"))

    for f_name in os.listdir("./scene"):
        if f_name[-3:] == '.py':
            copyfile(os.path.join("scene", f_name), os.path.join(cur_dir, f_name))

    arg_str = "Args: "
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        for i in range(len(sys.argv[1:])):
            if len(sys.argv[i + 1]) == 0:
                param = sys.argv[i + 1]
                arg_str += param + ", "
            elif sys.argv[i + 1][0] == "-":
                if i != 0:
                    arg_str += "; "
                arg = sys.argv[i + 1]
                arg_str += arg + "="
            else:
                param = sys.argv[i + 1]
                arg_str += param + ", "
        arg_str = arg_str.replace(", ;", ";")
        arg_str = arg_str.replace("=;", ";")
        cfg_log_f.write(str(Namespace(**vars(args))))
        cfg_log_f.write("\n")
        cfg_log_f.write(arg_str)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()

    parser = options.get_parser()

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Set training seed
    seed = 2024 * args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, batch=args.batch)

    # All done
    print("\nTraining complete.")
