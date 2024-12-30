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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from scene.deformable_field import Deformable_Field
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.visualizer_utils import feature_kmeans
import tinycudann as tcnn
from utils.camera_utils import quat_mul
from .deformable_field import positional_encoding
from pytorch3d.ops import knn_points
import frnn
from utils.fps import furthestsampling

class GaussianModel(nn.Module):
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.step_opacity_Function = lambda t, u: 1 / (1 + torch.exp(-(t - u) / self.beta))
        self.sharp_sigmoid = lambda t: 1 / (1 + torch.exp(-t / self.beta))


    def __init__(self, sh_degree : int, args):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.args = args
        self.beta = self.args.beta

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.N_pcd_init = None
        self.final_kpts_num = None

        self.super_gaussians = torch.empty(0)
        self.super_gaussians_feature = torch.empty(0)
        self.super_gaussians_weights = torch.empty(0)
        self.d = args.d
        self.w = args.w
        self.split_xyz = False

        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.motion_feature_dim = self.args.feature_dim
        self.radius_limit = 0.15
        
        self.No_prune_and_densify = ["s_xyz", "s_motion_feature", "s_weights", "weight_feature", "weight_mlp"]
        
        self.second_stage_iter = self.args.second_stage_iteration # 30000
        self.third_stage_iter = self.args.third_stage_iteration # 40000
        self.second_stage = False
        self.third_stage = False
        self.nearest_mask = None
        self.setup_functions()
        self.new_kpts_init()
    
    def restore(self, opt_dict, training_args, iteration=-1):
        self.training_args = training_args
        if iteration <= self.second_stage_iter:
            self.training_setup(training_args)
        elif iteration > self.third_stage_iter:
            self.training3stage_setup()
        else:
            self.training2stage_setup()
        self.optimizer.load_state_dict(opt_dict)

    def set_inputDim(self, time_input_dim, xyz_input_dim):
        self.time_input_dim = time_input_dim
        self.xyz_input_dim = xyz_input_dim
    
    @torch.no_grad()
    def get_nearest_mask(self, keepshape=False):
        if self.args.knn_type == "3D":
            _, nearest, _, _ = frnn.frnn_grid_points(self._xyz.unsqueeze(0), self.super_gaussians.unsqueeze(0), K=self.args.nearest_num, r=1e3)
        elif self.args.knn_type == "hybird":
            concate_xyz = torch.cat([self._xyz, self.motion_feature * self.args.feature_amplify], dim=-1)
            concate_super_xyz = torch.cat([self.super_gaussians, self.super_gaussians_feature * self.args.feature_amplify], dim=-1)
            _, nearest, _, _ = frnn.frnn_grid_points(concate_xyz.unsqueeze(0), concate_super_xyz.unsqueeze(0), K=self.args.nearest_num, r=1e8)
        else:
            print("Type error! Should be \"3D\" or \"hybird\" !!!! ")
            exit()

        if not keepshape:
            self.nearest_mask = nearest.view([-1])
        else:
            self.nearest_mask = nearest.view([self._xyz.shape[0], self.args.nearest_num])

    @torch.no_grad()
    def set_superKeypoints(self):
        print("Initialize Key Points")
        xyz = self.get_xyz
        feature = torch.cat([xyz, self.motion_feature], dim=-1)

        super_gaussians, super_features = feature_kmeans(xyz, feature, K=self.args.max_points)
        self.super_gaussians_feature = nn.Parameter(super_features[..., 3:].detach().requires_grad_(True))   
        self.super_gaussians = nn.Parameter(super_gaussians.detach().requires_grad_(True))
        self.training2stage_setup()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self): 
        return self._xyz
    
    @property
    def get_superGaussians(self):
        return self.super_gaussians
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_motion_feature(self):
        if self.second_stage:
            return self.super_gaussians_feature
        return self.motion_feature

    def new_kpts_init(self):
        self.new_xyz = None
        self.new_motion_feature = None
    
    def get_loss(self, iteration):
        if iteration < self.args.jointly_iteration:
            return 0.
        loss = 1.e-5 * torch.mean(torch.abs(self.get_motion_feature))
        return loss
    
    def get_motion_delta(self, xyz_embed, motion_feature, t):
        t = t.repeat((motion_feature.shape[0], 1))

        delta = self.df_model(torch.cat([motion_feature, xyz_embed, t], dim=-1))
        delta_xyz, delta_q = delta[..., 0:3], delta[..., 3:7]
        if self.args.step_opacity:
            delta_o = delta[..., 7:8]
        else:
            delta_o = None
        return delta_xyz, delta_q, delta_o
    
    def compute_dist(self):
        dist = self._xyz[:, None, :].repeat(1, self.super_gaussians.shape[0], 1) - \
               self.super_gaussians[None].repeat(self._xyz.shape[0], 1, 1)
        return torch.norm(dist, dim=-1)
    
    def get_new_kpts(self, mask, ratio=100):
        sampling = self.get_xyz[mask].contiguous()
        if sampling.shape[0] >= 1:
            select_num = sampling.shape[0] // ratio if sampling.shape[0] > ratio else 1
            clip = self.args.max_points + self.args.adaptive_points_num - self.super_gaussians.shape[0]
            select_num = select_num if select_num < clip else clip

            offset = torch.tensor([sampling.shape[0]], dtype=torch.int32).cuda()
            new_offset = torch.tensor([select_num], dtype=torch.int32).cuda()
            sample_idx = furthestsampling(sampling, offset, new_offset)
            new_xyz = sampling[sample_idx.to(torch.long)]

            _, idx, _ = knn_points(new_xyz.unsqueeze(0), self._xyz.unsqueeze(0), K=1)
            new_motion_feature = self.motion_feature[idx.view([-1])]
            self.new_xyz, self.new_motion_feature = new_xyz[:clip], new_motion_feature[:clip]
        else:
            self.new_xyz, self.new_motion_feature = None, None
    
    def fill_nearest(self, weights, nearest_mask=None, weights_fill=None):
        nearest_mask = self.nearest_mask if nearest_mask is None else nearest_mask
        weights_fill = self.weights_fill if weights_fill is None else weights_fill

        weights_xyz = weights[..., 0:self.args.nearest_num]
        weights_r = weights[..., self.args.nearest_num: self.args.nearest_num * 2]

        weights_xyz = nn.Softmax(dim=-1)(weights_xyz)
        weights_r = nn.Softmax(dim=-1)(weights_r)

        weights_xyz_inputs = weights_fill[..., :self.super_gaussians.shape[0]]
        weights_r_inputs = weights_fill[..., :self.super_gaussians.shape[0]]
        weights_xyz = torch.scatter(weights_xyz_inputs, -1, nearest_mask, weights_xyz)
        weights_r = torch.scatter(weights_r_inputs, -1, nearest_mask, weights_r)

        return weights_xyz, weights_r
    
    def forward(self, t, iteration, return_weights=False):
        t_input = t
        t = (positional_encoding(t, self.time_input_dim // 2)).unsqueeze(0)
        if torch.is_tensor(iteration):
            iteration = iteration.item()
        # Warm up stage
        if iteration < self.args.jointly_iteration:
            return self.get_xyz, self.get_rotation, self.get_scaling, self.get_opacity
        # First stage: Train a hyper canonical space for this dynamic scene
        if iteration <= self.second_stage_iter:
            decay_noise_xyz = torch.randn_like(self.get_xyz) * 0.1 * (1 - min(1, iteration / self.args.xyz_noise_iteration)) # Best Now.
            xyz = self.get_xyz.detach() + decay_noise_xyz
            motion_feature = self.motion_feature
            xyz_embed = positional_encoding(xyz, int(self.xyz_input_dim / 6))
        # Setting third stage training
        if iteration == self.third_stage_iter + 1:
            self.training3stage_setup()
        # Setting second stage training
        if iteration == self.second_stage_iter + 1: 
            self.set_superKeypoints()
        # Second stage: extract key points and deform with those key points
        if iteration > self.second_stage_iter:
            # decay_noise_gaussians = torch.randn_like(self.super_gaussians) * 0.1 * (2 - min(2, iteration / 20000)) # Best Now.
            decay_noise_gaussians = torch.randn_like(self.super_gaussians) * 0.1 * (1 - min(1, (iteration - self.second_stage_iter) / self.args.xyz_noise_iteration))
            xyz_embed = positional_encoding(self.get_superGaussians + decay_noise_gaussians , int(self.xyz_input_dim / 6))
            motion_feature = self.super_gaussians_feature
            weights = self.weights_model(self.get_xyz.detach())
            
            # Nearest weights
            self.get_nearest_mask(keepshape=True)
            weights_xyz, weights_r = self.fill_nearest(weights)
                
            self.weights_sum = torch.abs(weights_xyz) + torch.abs(weights_r)

        delta_xyz, delta_q, delta_o = self.get_motion_delta(xyz_embed, motion_feature, t)
        if self.args.norm_rotation:
            delta_q = self.rotation_activation(delta_q)

        self.kpts_xyz_motion = delta_xyz
        self.kpts_rotation_motion = delta_q
        if iteration > self.second_stage_iter:
            delta_q = weights_r @ delta_q
            delta_xyz = weights_xyz @ delta_xyz

            if self.args.densify_from_teaching:
                with torch.no_grad():
                    if iteration < self.args.adaptive_end_iter + self.second_stage_iter \
                        and iteration >= self.args.adaptive_from_iter + self.second_stage_iter\
                        and self.super_gaussians.shape[0] < self.args.max_points + self.args.adaptive_points_num:
                        self.get_teach_motion(t=t, delta_xyz=delta_xyz)
                        if iteration % self.args.adaptive_interval == 0:
                            mask = self.xyz_motion_accum_max.squeeze() >= self.args.teaching_threshold
                            self.get_new_kpts(mask)
        # compute current gaussian's position, rotation, scaling
        xyz = self.get_xyz + delta_xyz
        q = self.get_rotation_(self.rotation_activation(delta_q))
        s = self.get_scaling
        self.lifecycle_opacity = None
        
        # compute current gaussian's opacity
        if self.args.step_opacity and iteration > self.args.step_opacity_iteration:
            xyz_embed = positional_encoding(self.get_xyz, int(self.xyz_input_dim / 6))
            _, _, delta_o = self.get_motion_delta(xyz_embed, self.motion_feature, t)
            if self.args.opacity_type == "explicit":
                opacity = self.get_opacity * self.step_opacity_Function(t_input, self.opacity_thres)
            else:
                opacity = self.get_opacity * self.sharp_sigmoid(delta_o)
            self.lifecycle_opacity = opacity
            if return_weights and iteration > self.second_stage_iter:
                return xyz, q, s, self.get_opacity, weights_xyz, weights_r
            return xyz, q, s, opacity
        if return_weights and iteration > self.second_stage_iter:
            return xyz, q, s, self.get_opacity, weights_xyz, weights_r
        return xyz, q, s, self.get_opacity

    def get_teach_motion(self, t, delta_xyz):
        xyz_embed_teach = positional_encoding(self.get_xyz, int(self.xyz_input_dim / 6))

        motion_feature = self.motion_feature
        delta_xyz_teach, _, _ = self.get_motion_delta(xyz_embed_teach, motion_feature, t)
        norm = delta_xyz - delta_xyz_teach
        self.add_desification_stats_motion(norm)
    
    def get_rotation_(self, delta=0):
        return self.rotation_activation(quat_mul(delta, self._rotation))
    
    def get_scaling_(self, delta=0):
        return self.scaling_activation(self._scaling + delta)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        if self.N_pcd_init is not None:
            fused_point_cloud = fused_point_cloud[0:1].repeat(self.N_pcd_init, 1)
            fused_color = fused_color[0:1].repeat(self.N_pcd_init, 1)
        self.N_pcd_init = fused_point_cloud.shape[0]
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # step opacity threshold
        delta_dim = 7
        if self.args.step_opacity:
            self.opacity_thres = nn.Parameter(-2 * torch.ones_like(opacities).requires_grad_(True))
            delta_dim += 1

        self.motion_feature = nn.Parameter(1e-3 * (2 * torch.rand((self.N_pcd_init, self.motion_feature_dim)).cuda() - 1))
        self.df_model = Deformable_Field(self.time_input_dim + self.xyz_input_dim + self.motion_feature_dim, d=self.d, w=self.w, output_dim=delta_dim, split_xyz=self.split_xyz).to('cuda')
        if self.final_kpts_num is not None:
            self.super_gaussians = nn.Parameter(torch.ones([self.final_kpts_num, 3], dtype=torch.float32).requires_grad_(True).to("cuda"))
            self.super_gaussians_feature = nn.Parameter(torch.ones([self.final_kpts_num, self.motion_feature_dim], dtype=torch.float32).requires_grad_(True).to("cuda"))
        else:
            self.super_gaussians = nn.Parameter(torch.ones([self.args.max_points, 3], dtype=torch.float32).requires_grad_(True).to("cuda"))
            self.super_gaussians_feature = nn.Parameter(torch.ones([self.args.max_points, self.motion_feature_dim], dtype=torch.float32).requires_grad_(True).to("cuda"))
        k=2; L = 16; F = 4; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        self.weights_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=k * self.args.nearest_num,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
      
    def training3stage_setup(self):
        self.third_stage = True
        self.weights_fill = torch.zeros([self._xyz.shape[0], self.args.max_points + self.args.adaptive_points_num], dtype=torch.float32).cuda()
        l = [{'params': [self._xyz], 'lr': self.training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': self.training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': self.training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.training_args.rotation_lr, "name": "rotation"},
            {'params': [self.super_gaussians], 'lr': self.training_args.kpts_lr, "name": "s_xyz"},
            {'params': [self.super_gaussians_feature], 'lr': self.training_args.kpts_lr, "name": "s_motion_feature"},
            {'params': self.weights_model.parameters(), 'lr': self.training_args.hash_lr, "name": "weight_mlp"},
            {'params': self.df_model.parameters(), 'lr': self.training_args.mlp_lr, "name": "df_mlp"}
            ]
        if self.args.step_opacity:
            l.append({'params': [self.opacity_thres], 'lr': self.training_args.opacity_lr, "name": "opacity_thres"})
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def training2stage_setup(self):
        self.second_stage = True   
        self.xyz_motion_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_motion_accum_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.motion_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.kpts_gradient_accum = torch.zeros((self.super_gaussians.shape[0], 1), device="cuda")
        self.kpts_gradient_accum_max = torch.zeros((self.super_gaussians.shape[0], 1), device="cuda")
        self.kpts_denom = torch.zeros((self.super_gaussians.shape[0], 1), device="cuda")
        self.register_buffer("weights_fill", torch.zeros([self._xyz.shape[0], self.args.max_points + self.args.adaptive_points_num], dtype=torch.float32).cuda())
        
        l = [
            {'params': [self.super_gaussians], 'lr': self.training_args.kpts_lr, "name": "s_xyz"},
            {'params': [self.super_gaussians_feature], 'lr': self.training_args.kpts_lr, "name": "s_motion_feature"},
            {'params': self.weights_model.parameters(), 'lr': self.training_args.hash_lr, "name": "weight_mlp"},
            {'params': self.df_model.parameters(), 'lr': self.training_args.mlp_lr, "name": "df_mlp"}
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def training_setup(self, training_args):
        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.kpts_gradient_accum = torch.zeros((self.args.max_points, 1), device="cuda")
        self.kpts_gradient_accum_max = torch.zeros((self.args.max_points, 1), device="cuda")
        self.kpts_denom = torch.zeros((self.args.max_points, 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.df_model.parameters(), 'lr': training_args.mlp_lr, "name": "df_mlp"},
            {'params': [self.motion_feature], 'lr': training_args.mfeature_lr, "name": "motion_feature"}
        ]
        if self.args.step_opacity:
            l.append({'params': [self.opacity_thres], 'lr': training_args.opacity_lr, "name": "opacity_thres"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=self.training_args.mlp_lr,
                                                    lr_final=training_args.position_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.weight_mlp_scheduler_args = get_expon_lr_func(lr_init=self.training_args.hash_lr,
                                                    lr_final=self.training_args.hash_lr_final,
                                                    lr_delay_steps=training_args.position_lr_max_steps,
                                                    max_steps=training_args.iterations) # not in args, hard code
        self.motion_feature_scheduler_args = get_expon_lr_func(lr_init=self.training_args.mfeature_lr,
                                                    lr_final=self.training_args.mfeature_lr_final,
                                                    lr_delay_steps=training_args.position_lr_max_steps,
                                                    max_steps=training_args.position_lr_max_steps)
        self.super_xyz_scheduler_args = get_expon_lr_func(lr_init=self.training_args.kpts_lr,
                                                    lr_final=self.training_args.kpts_lr_final,
                                                    lr_delay_steps=training_args.position_lr_max_steps,
                                                    max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz" or "delta" in param_group["name"] or "fourier_weights" in param_group["name"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif "mlp" in param_group["name"] and "weight" not in param_group["name"]:
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "s_xyz" or param_group["name"] == "weight_feature":
                lr = self.super_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif "weight_mlp" in param_group["name"]:
                lr = self.weight_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif "motion_feature" in param_group["name"] or "motion_weights"  in param_group["name"] or param_group["name"] == "delta_xyz":
                lr = self.motion_feature_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" in optimizable_tensors.keys():
            self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1 or group["name"] in self.No_prune_and_densify:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        key = optimizable_tensors.keys()
        self._xyz = optimizable_tensors["xyz"] if "xyz" in key else nn.Parameter(self._xyz[valid_points_mask])
        self._features_dc = optimizable_tensors["f_dc"] if "f_dc" in key else nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = optimizable_tensors["f_rest"] if "f_rest" in key else nn.Parameter(self._features_rest[valid_points_mask])
        self._opacity = optimizable_tensors["opacity"] if "opacity" in key else nn.Parameter(self._opacity[valid_points_mask])
        self._scaling = optimizable_tensors["scaling"] if "scaling" in key else nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = optimizable_tensors["rotation"] if "rotation" in key else nn.Parameter(self._rotation[valid_points_mask])
        self.motion_feature = optimizable_tensors["motion_feature"] if "motion_feature" in key else nn.Parameter(self.motion_feature[valid_points_mask])
        if self.args.step_opacity:
            self.opacity_thres = optimizable_tensors["opacity_thres"] if "opacity_thres" in key else nn.Parameter(self.opacity_thres[valid_points_mask])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_max = self.xyz_gradient_accum_max[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, use_No_prune=True):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1 or (group["name"] in self.No_prune_and_densify and use_No_prune):
                continue
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_motion_postfix(self, new_xyz, new_motion_feature):
        d = {"s_xyz": new_xyz,
        "s_motion_feature": new_motion_feature}

        optimizable_tensors = self.cat_tensors_to_optimizer(d, use_No_prune=False)
        self.super_gaussians = optimizable_tensors["s_xyz"]
        self.super_gaussians_feature = optimizable_tensors["s_motion_feature"]

        self.xyz_motion_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_motion_accum_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.motion_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.kpts_gradient_accum = torch.zeros((self.super_gaussians.shape[0], 1), device="cuda")
        self.kpts_gradient_accum_max = torch.zeros((self.super_gaussians.shape[0], 1), device="cuda")
        self.kpts_denom = torch.zeros((self.super_gaussians.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                              new_motion_feature=None, new_opacity_thres=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "motion_feature": new_motion_feature}
        if self.args.step_opacity:
            d["opacity_thres"] = new_opacity_thres

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        key = optimizable_tensors.keys()
        self._xyz = optimizable_tensors["xyz"] if "xyz" in key else nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0))
        self._features_dc = optimizable_tensors["f_dc"] if "f_dc" in key else nn.Parameter(torch.cat((self._features_dc, new_features_dc), dim=0))
        self._features_rest = optimizable_tensors["f_rest"] if "f_rest" in key else nn.Parameter(torch.cat((self._features_rest, new_features_rest), dim=0))
        self._opacity = optimizable_tensors["opacity"] if "opacity" in key else nn.Parameter(torch.cat((self._opacity, new_opacities), dim=0))
        self._scaling = optimizable_tensors["scaling"] if "scaling" in key else nn.Parameter(torch.cat((self._scaling, new_scaling), dim=0))
        self._rotation = optimizable_tensors["rotation"] if "rotation" in key else nn.Parameter(torch.cat((self._rotation, new_rotation), dim=0))
        self.motion_feature = optimizable_tensors["motion_feature"] if "motion_feature" in key else nn.Parameter(torch.cat((self.motion_feature, new_motion_feature), dim=0))
        if self.args.step_opacity:
            self.opacity_thres = optimizable_tensors["opacity_thres"] if "opacity_thres" in key else nn.Parameter(torch.cat((self.opacity_thres, new_opacity_thres), dim=0))


        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_motion_feature = self.motion_feature[selected_pts_mask].repeat(N,1)
        new_opacity_thres = None
        if self.args.step_opacity:
            new_opacity_thres = self.opacity_thres[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_motion_feature=new_motion_feature,
                                   new_opacity_thres=new_opacity_thres)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_motion_feature = self.motion_feature[selected_pts_mask]
        new_opacity_thres = None
        if self.args.step_opacity:
            new_opacity_thres = self.opacity_thres[selected_pts_mask]
    
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_motion_feature=new_motion_feature,
                                   new_opacity_thres=new_opacity_thres)

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        torch.cuda.empty_cache()

    def densify_kpts(self, max_grad, mode="gaussian_mean", ratio=100):
        if mode == "down_sampling":
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            mask = (grads > max_grad).squeeze()
            self.get_new_kpts(mask, ratio=ratio)
        else:
            if mode == "gaussian_mean":
                grads = self.xyz_gradient_accum / self.denom
                grads[grads.isnan()] = 0.0
                mask = (grads > max_grad).squeeze()
                weights_value, index = self.weights_sum[mask, :self.super_gaussians.shape[0]].max(dim=-1)
                clone_idx = index.unique()
            else:
                grads = self.kpts_gradient_accum / self.kpts_denom
                grads[grads.isnan()] = 0.0
                clone_idx = (grads >= max_grad).view([-1])
            clip = self.args.max_points + self.args.adaptive_points_num - self.super_gaussians.shape[0]
            self.new_xyz = self.super_gaussians[clone_idx][:clip]
            self.new_motion_feature = self.super_gaussians_feature[clone_idx][:clip]
        
        if self.new_xyz is not None:
            self.densification_motion_postfix(self.new_xyz, self.new_motion_feature)
            self.new_kpts_init()
        torch.cuda.empty_cache()
    
    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        grad = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += grad
        self.denom[update_filter] += 1
        self.xyz_gradient_accum_max[update_filter] = torch.where(grad > self.xyz_gradient_accum_max[update_filter], grad, self.xyz_gradient_accum_max[update_filter])

    def add_desification_stats_motion(self, viewspace_point_tensor, update_filter=None):
        if update_filter is None:
            motion = torch.norm(viewspace_point_tensor, dim=-1, keepdim=True)
            # self.xyz_motion_accum += motion
            self.xyz_motion_accum_max = torch.where(motion > self.xyz_motion_accum_max, motion, self.xyz_motion_accum_max)
            self.motion_denom += 1
        else:
            grad = torch.norm(viewspace_point_tensor.grad, dim=-1, keepdim=True)
            self.kpts_gradient_accum += grad
            self.kpts_gradient_accum_max = torch.where(grad > self.kpts_gradient_accum_max, grad, self.kpts_gradient_accum_max)
            self.kpts_denom += 1