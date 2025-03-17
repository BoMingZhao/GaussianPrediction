import torch
import torch.nn.functional as F
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from utils.prepare.makeVideo import make_video
from metrics import evaluate
from options.gaussian_option import Gaussian_Options
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.visualizer_utils import *
from motion_model.gcn import GCN_xyzr, get_dct_matrix
from motion_model.dataset import GCN3DDataset
from gaussian_renderer import GaussianModel
import imageio
from eval import render_kpts

def operate(args, batch, model, eval=False, noise_xyz=0, noise_r=0):
        if eval:
            model.eval()
        xyz_inputs = batch["xyz_inputs"]
        xyz_gt = batch["xyz_gt"]
        r_inputs, r_gt = batch["rotation_inputs"], batch["rotation_gt"]

        if noise_xyz > 0:
            xyz_inputs_noise = (2*torch.rand_like(xyz_inputs)-1) * noise_xyz
            xyz_inputs = xyz_inputs + xyz_inputs_noise
            
        if noise_r > 0:
            r_inputs_noise = (2*torch.rand_like(r_inputs)-1) * noise_r
            r_inputs = r_inputs + r_inputs_noise
            if args.norm_rotation:
                r_inputs = F.normalize(r_inputs, dim=-1)

        xyz_pred, r_pred = model(xyz_inputs.permute((0, 3, 2, 1)), r_inputs.permute((0, 3, 2, 1)))
        xyz_pred = xyz_pred.permute((0, 3, 2, 1))
        r_pred = r_pred.permute((0, 3, 2, 1))
        if args.norm_rotation:
            r_pred = F.normalize(r_pred, dim=-1)
        if eval:
            model.train()
        return xyz_pred, xyz_gt, r_pred, r_gt

def video(model_path, name, iteration, fps=10):
    image_dir = os.path.join(model_path + "kpts", name, "ours_{}".format(iteration), "renders")
    images = []
    for i in range(len(os.listdir(image_dir))):
        if os.path.exists(os.path.join(image_dir, f"{i:05d}.png")):
            image = imageio.imread(os.path.join(image_dir, f"{i:05d}.png"))
            images.append(image)
    images = np.stack(images, axis=0)
    imageio.mimwrite(os.path.join(image_dir, f"video.mp4"), images, fps=fps)

def train_gcn(args, opt, gaussians, scene, pipeline, background, iteration=60000, predict_more=False, metrics=False):
    with torch.no_grad():
        train_dataset = GCN3DDataset(gaussians, opt.time_freq, iteration=iteration, model_path=args.model_path, source_path=args.source_path,
                                    max_time=args.max_time, input_size=args.input_size, output_size=args.output_size, split="train")
        test_dataset = GCN3DDataset(gaussians, opt.time_freq, iteration=iteration, model_path=args.model_path, source_path=args.source_path,
                                    max_time=args.max_time, input_size=args.input_size, output_size=args.output_size, split="test")
    trainLoarder = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=0)
    
    
    dct_n = args.input_size
    dct, idct = get_dct_matrix(dct_n)
    dct = torch.from_numpy(dct.astype(np.float32)).cuda()
    idct = torch.from_numpy(idct.astype(np.float32)).cuda()

    model = GCN_xyzr(input_feature=args.input_size, hidden_feature=args.linear_size, output_feature=args.output_size, 
                p_dropout=args.dropout, num_stage=args.num_stage, node_n=train_dataset.nodes_num, no_mapping=args.no_mapping).cuda()
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-15)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1.e-4)
    # t_bar = tqdm(total=args.epoch*len(trainLoarder))
    if args.evaluate:
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, args.exp_name, "ckpt.pth")))
        except:
            error_path = os.path.join(args.model_path, args.exp_name, "ckpt.pth")
            raise Exception(f"Can not load checkpoint of GCN from {error_path}.")

    else:
        t_bar = tqdm(total=args.epoch)
        t_bar.set_description('[Train GCN]')
        for epoch in range(args.epoch):
            loss_avg = 0.
            for idx, batch in enumerate(trainLoarder):
                noise_xyz = 0.
                noise_r = 0.
                if args.noise_init > 0:
                    noise_xyz = args.noise_init * max(1.-epoch/args.noise_step, 0.)
                    noise_r = args.noise_init * max((1.-epoch / args.noise_step), 0.) * 0.5

                xyz_pred, xyz_gt, r_pred, r_gt = operate(args, batch, model, noise_xyz=noise_xyz, noise_r=noise_r)
                loss = torch.mean(torch.norm(xyz_pred - xyz_gt, 2, -1)) + torch.mean(torch.norm(r_pred - r_gt, 2, -1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_avg += loss
            
            t_bar.set_postfix(loss=loss_avg.item() / (idx + 1), lr=optimizer.param_groups[0]['lr'], noise_xyz=noise_xyz, noise_r=noise_r)
            t_bar.update(1)
            scheduler.step()
        
        os.makedirs(os.path.join(args.model_path, args.exp_name), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.model_path, args.exp_name, "ckpt.pth"))

    batch = test_dataset[0]
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0)
    xyz_inputs_cache = batch["xyz_inputs"].clone()
    xyz_gt_cache = batch["xyz_gt"].clone()
    rotation_inputs_cache = batch["rotation_inputs"].clone()
    rotation_gt_cache = batch["rotation_gt"].clone()
    if predict_more:
        frames = args.frames
        with torch.no_grad():
            kpts = []
            kpts_rotation = []
            views = scene.getTestCameras()[args.cam_id:]

            # here we take prediction of GCN model as input
            t_bar = tqdm(total=frames)
            for _ in range(frames):
                xyz_pred, xyz_gt, r_pred, r_gt = operate(args, batch, model, eval=True)
                kpts += [xyz_pred[0][-args.output_size:, ...]]
                kpts_rotation += [r_pred[0][-args.output_size:, ...]]
                batch["xyz_inputs"] = torch.cat([batch["xyz_inputs"][:, args.output_size:], xyz_pred[:, -args.output_size:, ...]], dim=1)
                batch["rotation_inputs"] = torch.cat([batch["rotation_inputs"][:, args.output_size:], r_pred[:, -args.output_size:, ...]], dim=1)
                t_bar.set_description('[Predicted by GCN]')
                t_bar.update(1)

            kpts = torch.cat(kpts, dim=0)
            kpts_rotation = torch.cat(kpts_rotation, dim=0)
            _, render_path = render_kpts(args.model_path, args.exp_name, "Predicted_more_by_GCN", iteration, views, gaussians, pipeline, background, 
                                      kpts, kpts_rotation, view_id=args.cam_id)
            print("Generate Video!")
            output_name = os.path.basename(render_path) + ".mp4"
            make_video(render_path, image_num=args.frames, fps=30, concat_delta=False, output_name=output_name)
            
    if metrics:
        with torch.no_grad():
            batch = test_dataset[0]
            batch["xyz_inputs"] = xyz_inputs_cache
            batch["xyz_gt"] = xyz_gt_cache
            batch["rotation_inputs"] = rotation_inputs_cache
            batch["rotation_gt"] = rotation_gt_cache
            kpts, kpts_rotation = [], []
            views = scene.getTestCameras()
            frames = len(views)
            t_bar = tqdm(total=frames)
            time_interval = test_dataset.test_times[1] - test_dataset.test_times[0]
            time = test_dataset.test_times[0]
            print(f"Predict time from: {time}, Time interval: {time_interval}")

            for _ in range(frames):
                xyz_pred, xyz_gt, r_pred, r_gt = operate(args, batch, model, eval=True)
                kpts += [xyz_pred[0][-args.output_size:, ...]]
                kpts_rotation += [r_pred[0][-args.output_size:, ...]]
                batch["xyz_inputs"] = torch.cat([batch["xyz_inputs"][:, args.output_size:], xyz_pred[:, -args.output_size:, ...]], dim=1)
                batch["rotation_inputs"] = torch.cat([batch["rotation_inputs"][:, args.output_size:], r_pred[:, -args.output_size:, ...]], dim=1)
                t_bar.set_description('[Directly predicted by GCN]')
                t_bar.set_postfix(time=time)
                t_bar.update(1)
                time += time_interval

            kpts = torch.cat(kpts, dim=0)
            kpts_rotation = torch.cat(kpts_rotation, dim=0)
            eval_path, _ = render_kpts(args.model_path, args.exp_name, "[metrics]Predicted_by_GCN_on_test_views", iteration, views, gaussians, pipeline, 
                                    background, kpts, kpts_rotation, metrics=metrics)
            evaluate(eval_path, resize_ratio=args.resize)
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training gcn script parameters")
    op = OptimizationParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ckpt_iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser_options = Gaussian_Options(parser)
    parser_options.initial()
    parser_options.gcn_training()
    parser = parser_options.get_parser()
    
    args = get_combined_args(parser)
    args.dct_n = args.input_size + args.output_size
    print("Training GCN " + args.model_path)

    # Prepare Gaussian model
    dataset = model.extract(args)
    opt = op.extract(args)
    pipeline = pipeline.extract(args)

    xyz_freq = opt.xyz_freq
    time_freq = opt.time_freq

    time_input_dim = 2 * time_freq
    xyz_input_dim = 6 * xyz_freq

    gaussians = GaussianModel(dataset.sh_degree, args)
    gaussians.set_inputDim(time_input_dim, xyz_input_dim)
    (model_params, opt_dict, first_iter) = torch.load(os.path.join(dataset.model_path, f"chkpnt{args.ckpt_iteration:d}.pth"))
    gaussians.N_pcd_init = model_params["_xyz"].shape[0]
    gaussians.active_sh_degree = gaussians.max_sh_degree
    gaussians.final_kpts_num = model_params["super_gaussians"].shape[0] if "super_gaussians" in model_params.keys() else None

    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.restore(opt_dict, opt, first_iter)
    gaussians.load_state_dict(model_params, strict=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_gcn(args, opt, gaussians, scene, pipeline, background, iteration=args.ckpt_iteration, predict_more=args.predict_more, metrics=args.metrics)
