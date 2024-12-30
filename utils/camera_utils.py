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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

WARNED = False
def lerp(t, v0, v1, is_tensor):
    x = (1 - t) * v0 + t * v1
    if is_tensor:
        x = torch.from_numpy(x).cuda()
    return x

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy, c)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    if c:
        res = torch.from_numpy(v2).to("cuda")
    else:
        res = v2
    return res

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    fwd_flow = None
    fwd_flow_mask = None
    bwd_flow = None
    bwd_flow_mask = None
    if "fwd_flow" in cam_info._fields:
        fwd_flow = cam_info.fwd_flow
        fwd_flow_mask = cam_info.fwd_flow_mask
        bwd_flow = cam_info.bwd_flow
        bwd_flow_mask = cam_info.bwd_flow_mask
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  time=cam_info.time, fwd_flow=fwd_flow, bwd_flow=bwd_flow, fwd_flow_mask=fwd_flow_mask, bwd_flow_mask=bwd_flow_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        if isinstance(c, dict):
            cam = loadCam(args, id, c["cam"], resolution_scale)
            fwd_cam = None
            bwd_cam = None
            if c["fwd_cam"] is not None:
                fwd_cam = loadCam(args, id, c["fwd_cam"], resolution_scale)
            if c["bwd_cam"] is not None:
                bwd_cam = loadCam(args, id, c["bwd_cam"], resolution_scale)
            data = {"cam": cam, "fwd_cam": fwd_cam, "bwd_cam": bwd_cam}
            camera_list.append(data)
        else:
            camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def quat_mul(q1, q2):
    '''
    all quaternion in (w, x, y, z)
    q1: (N, 4)
    q2: (N, 4)
    return q1q2
    '''
    q1q2 = torch.zeros_like(q1)
    q1q2[..., 0] = q2[..., 0]*q1[..., 0] - q2[..., 1]*q1[..., 1] - q2[..., 2]*q1[..., 2] - q2[..., 3]*q1[..., 3]
    q1q2[..., 1] = q2[..., 1]*q1[..., 0] + q2[..., 0]*q1[..., 1] + q2[..., 3]*q1[..., 2] - q2[..., 2]*q1[..., 3]
    q1q2[..., 2] = q2[..., 2]*q1[..., 0] - q2[..., 3]*q1[..., 1] + q2[..., 0]*q1[..., 2] + q2[..., 1]*q1[..., 3]
    q1q2[..., 3] = q2[..., 3]*q1[..., 0] + q2[..., 2]*q1[..., 1] - q2[..., 1]*q1[..., 2] + q2[..., 0]*q1[..., 3]
    return q1q2

def quat_to_axis_theta(q):
    theta = 2 * torch.arccos(q[0])
    v = 1 / torch.sqrt(1 - q[0]**2)
    v = v * q[1:]
    return v, theta

def axis_theta_to_quat(axis, theta):
    w = torch.cos(theta / 2).view([1])
    xyz = axis * torch.sqrt(1 - w**2)

    return torch.cat([w, xyz.view([-1])], dim=0)

def slerp_tensor(q0, q1, t):
    dot = (q0 * q1).sum(dim=-1)
    theta = torch.acos(dot)

    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / (sin_theta + 1e-8)
    s1 = torch.sin(t * theta) / (sin_theta + 1e-8)

    q = s0[..., None] * q0 + s1[..., None] * q1
    return q

def transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (N, 3, 3) or (3, 3)
    :param t: translation vector could be (N, 3, 1) or (3, 1)
    :param X: points with 3D position, a 2D array with dimension of (N, num_points, 3) or (num_points, 3)
    :return: transformed 3D points
    """
    keep_dim_n = False
    keep_dim_hw = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    if X.dim() == 2:
        X = X.unsqueeze(0)

    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    N = R.shape[0]
    M = X.shape[1]
    X_after_R = torch.bmm(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)

    if keep_dim_hw:
        trans_X = trans_X.view(N, H, W, 3)
    if keep_dim_n:
        trans_X = trans_X.squeeze(0)

    return trans_X

def pi(K, X):
    """
    Projecting the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix tensor (N, 3, 3) or (3, 3)
    :param X: point position in 3D camera coordinates system, is a 3D array with dimension of (N, num_points, 3), or (num_points, 3)
    :return: N projected 2D pixel position u (N, num_points, 2) and the depth X (N, num_points, 1)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if X.dim() == 2:
        X = X.unsqueeze(0)      # make dim (1, num_points, 3)
    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    assert K.size(0) == X.size(0)
    N = K.shape[0]

    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    u_x = fx * X[:, :, 0:1] / (X[:, :, 2:3] + 0.0001) + cx
    u_y = fy * X[:, :, 1:2] / (X[:, :, 2:3] + 0.0001) + cy
    u = torch.cat([u_x, u_y], dim=-1)
    d = X[:, :, 2:3]

    if keep_dim_hw:
        u = u.view(N, H, W, 2)
        d = d.view(N, H, W)
    if keep_dim_n:
        u = u.squeeze(0)
        d = d.squeeze(0)

    return u, d

def interpolation_pose(view, previous_view, interpolation_ratio):
    t, pre_t, R, pre_R = view.T, previous_view.T, view.R, previous_view.R
    new_t = pre_t + (t - pre_t) * interpolation_ratio
    quat = matrix_to_quaternion(torch.from_numpy(R))
    pre_quat = matrix_to_quaternion(torch.from_numpy(pre_R))
    new_quat = slerp(interpolation_ratio, pre_quat, quat)
    new_R = quaternion_to_matrix(new_quat[None]).squeeze().cpu().numpy()
    return new_t, new_R