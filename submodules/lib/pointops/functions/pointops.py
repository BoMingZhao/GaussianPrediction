from enum import unique
from hashlib import new
from math import exp, floor
from sys import stderr
from typing import Tuple
from sklearn.metrics import det_curve

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops_cuda
import frnn
import time
from plyfile import PlyData, PlyElement
import numpy as np

from util import glo
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

use_unaligned_knn = False



class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

furthestsampling = FurthestSampling.apply

class FarthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max, m_max = xyz.shape[0], offset.shape[0], offset[0], new_offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
            m_max = max(new_offset[i] - new_offset[i-1], m_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        # print(f"n_max: {n_max}, b: {b}", flush=True)
        
        pointops_cuda.farthestsampling_cuda(b, n_max, m_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

farthestsampling = FarthestSampling.apply

def random_sample(xyz, offset, new_offset):
    """
    input: xyz: (n, 3), offset: (b), new_offset: (b)
    output: idx: (m)
    """
    assert xyz.is_contiguous()
    idx = []
    for i in range(new_offset.size(0)):
        if i == 0:
            idx.append(torch.randperm(offset[i], device=torch.device("cuda"))[:new_offset[i]])
        else:
            idx.append(torch.randperm(offset[i] - offset[i - 1], device=torch.device("cuda"))[:new_offset[i] - new_offset[i - 1]] + new_offset[i - 1])
    return torch.cat(idx)

def fnv_hash_vec(arr):
    """
    FNV-1
    64bit:
        offset_basis: (unsigned) 14695981039346656037 signed -3750763034362895579
        prime: 1099511628211
    32bit:
        offset_basis: 2166136261
        prime: 16777619
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.long()
    hashed_arr = torch.tensor([-3750763034362895579], dtype=torch.long, device=torch.device("cuda")) * torch.ones(arr.size(0), dtype=torch.long, device=torch.device("cuda"))
    for j in range(arr.size(1)):
        hashed_arr *= torch.tensor([1099511628211], dtype=torch.long, device=torch.device("cuda"))
        hashed_arr = torch.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def voxelize(coord, voxel_size=0.05, hash_type='fnv'):
    discrete_coord = torch.floor(coord / torch.tensor(voxel_size))

    key = fnv_hash_vec(discrete_coord)

    # idx_sort = torch.argsort(key)
    # key_sort = key[idx_sort]
    key_sort, idx_sort = torch.sort(key, dim=0)
    _, count = torch.unique_consecutive(key_sort, return_counts=True, dim=0)
    idx_select = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int64, device=torch.device("cuda")), count))[0:-1], dim=0) + torch.randint(0, count.max(), (count.size(0), ), device=torch.device("cuda")) % count
    # idx_select = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int64, device=torch.device("cuda")), count))[0:-1], dim=0)
    idx_unique = idx_sort[idx_select]
    return idx_unique

def voxel_sample(xyz, offset, new_offset, voxel_size):
    """
    input: xyz: (n, 3), offset: (b), new_offset: (b)
    output: idx: (m)
    """
    assert xyz.is_contiguous()
    idx = []
    for i in range(new_offset.size(0)):
        coord = xyz[offset[i - 1]:offset[i]] if i != 0 else xyz[:offset[0]]
        coord_min = torch.min(coord, 0)[0]
        coord = coord - coord_min
        uniq_idx = voxelize(coord, voxel_size)
        cnt = offset[i] - offset[i - 1] if i != 0 else offset[0]
        expect_cnt = new_offset[i] - new_offset[i - 1] if i != 0 else new_offset[0]
        sample_cnt = uniq_idx.size(0)

        # print(f"voxel: {voxel_size}\texpect cnt: {expect_cnt}\tsample cnt: {sample_cnt}", file=stderr)
        if sample_cnt > expect_cnt:
            coord = coord[uniq_idx]
            revise_idx = uniq_idx[torch.randperm(sample_cnt, device=torch.device("cuda"))[:expect_cnt]]
            # init_idx = torch.randint(uniq_idx.size(0), (1,))
            # revise_idx = uniq_idx[torch.argsort(torch.sum(torch.square(coord - coord[init_idx]), 1), descending=True)[:expect_cnt]]
        elif sample_cnt < expect_cnt:
            revise_idx = torch.cat((uniq_idx, torch.randperm(cnt, device=torch.device("cuda"))[:expect_cnt - sample_cnt]))
        else:
            revise_idx = uniq_idx
        # revise_idx = uniq_idx
        if i != 0:
            revise_idx += new_offset[i - 1]
        idx.append(revise_idx)
    return torch.cat(idx)

def voxelize_ratio(coord, voxel_size=0.05, ratio = 0.25, fps = True):
    discrete_coord = torch.floor(coord / torch.tensor(voxel_size))

    key = fnv_hash_vec(discrete_coord)

    # idx_sort = torch.argsort(key)
    # key_sort = key[idx_sort]
    key_sort, idx_sort = torch.sort(key, dim=0)
    _, count = torch.unique_consecutive(key_sort, return_counts=True, dim=0)

    count_select = torch.floor(count * ratio).int()
    if fps:
        offset = torch.cumsum(count, dim=0, dtype=torch.int32)
        new_offset = torch.cumsum(count_select, dim=0, dtype=torch.int32)
        idx_select = furthestsampling(coord[idx_sort], offset, new_offset).long()
    else: # random select in grid
        # TODO: use CUDA to parallelize
        idx_base = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int64, device=torch.device("cuda")), count))[0:-1], dim=0) # first point in grid
        idx_select = []
        for i in range(count.size(0)):
            idx_ratio = torch.randperm(count[i], device=torch.device("cuda"))[:count_select[i]]
            idx_select.append(idx_ratio + idx_base[i])
        idx_select = torch.cat(idx_select)
    # idx_select = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int64, device=torch.device("cuda")), count))[0:-1], dim=0) + torch.randint(0, count.max(), (count.size(0), ), device=torch.device("cuda")) % count
    # idx_select = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int64, device=torch.device("cuda")), count))[0:-1], dim=0)
    idx_unique = idx_sort[idx_select]
    return idx_unique

def voxel_sample_ratio(xyz, offset, new_offset, voxel_size, ratio=0.25, fps = True):
    """
    input: xyz: (n, 3), offset: (b), new_offset: (b)
    output: idx: (m)
    """
    assert xyz.is_contiguous()
    idx = []
    for i in range(new_offset.size(0)):
        coord = xyz[offset[i - 1]:offset[i]] if i != 0 else xyz[:offset[0]]
        coord_min = torch.min(coord, 0)[0]
        coord = coord - coord_min
        uniq_idx = voxelize_ratio(coord, voxel_size, ratio, fps)
        cnt = offset[i] - offset[i - 1] if i != 0 else offset[0]
        expect_cnt = new_offset[i] - new_offset[i - 1] if i != 0 else new_offset[0]
        sample_cnt = uniq_idx.size(0)

        # print(f"voxel: {voxel_size}\texpect cnt: {expect_cnt}\tsample cnt: {sample_cnt}", file=stderr)
        if sample_cnt > expect_cnt:
            raise Exception("ratio implement error!")
        elif sample_cnt < expect_cnt:
            revise_idx = torch.cat((uniq_idx, torch.randperm(cnt, device=torch.device("cuda"))[:expect_cnt - sample_cnt]))
        else:
            revise_idx = uniq_idx
        if i != 0:
            revise_idx += new_offset[i - 1]
        idx.append(revise_idx)
    return torch.cat(idx)


def voxelize_fps_ratio(coord, voxel_size=0.05, expect_num = 1000, ratio = 0.25):
    discrete_coord = torch.floor(coord / torch.tensor(voxel_size))

    key = fnv_hash_vec(discrete_coord)

    key_sort, idx_sort = torch.sort(key, dim=0)
    _, count = torch.unique_consecutive(key_sort, return_counts=True, dim=0)

    total_num = coord.size(0)
    grid_num = count.size(0)
    
    if grid_num >= expect_num:
        return True, _

    revise_ratio = (expect_num - grid_num) / total_num
    count_select = torch.floor(((count - 1) * revise_ratio) + 1).int()

    grid_offset = torch.cumsum(count, dim=0, dtype=torch.int32)
    grid_new_offset = torch.cumsum(count_select, dim=0, dtype=torch.int32)
    idx_select = furthestsampling(coord[idx_sort], grid_offset, grid_new_offset).long()
    
    idx_unique = idx_sort[idx_select]
    # print(f"total_num: {total_num}, grid_num: {grid_num}, expect_num: {expect_num}, true select num: {idx_unique.size(0)} ratio: {revise_ratio}", file=stderr)
    return False, idx_unique


def fps_in_voxel_sample(xyz, offset, new_offset, voxel_size, ratio=0.25):
    """
    input: xyz: (n, 3), offset: (b), new_offset: (b)
    output: idx: (m)
    """
    assert xyz.is_contiguous()
    idx = []
    for i in range(new_offset.size(0)):
        coord = xyz[offset[i - 1]:offset[i]] if i != 0 else xyz[:offset[0]]
        coord_min = torch.min(coord, 0)[0]
        coord = coord - coord_min
        
        cnt = offset[i] - offset[i - 1] if i != 0 else offset[0]
        expect_cnt = new_offset[i] - new_offset[i - 1] if i != 0 else new_offset[0]
        compl_fps, uniq_idx = voxelize_fps_ratio(coord, voxel_size, expect_cnt, ratio)
        if compl_fps: # TODO: simplify
            uniq_idx = furthestsampling(coord, cnt.unsqueeze(0), expect_cnt.unsqueeze(0))
            print("use complete fps", file=stderr)
        sample_cnt = uniq_idx.size(0)

        # print(f"voxel: {voxel_size}\texpect cnt: {expect_cnt}\tsample cnt: {sample_cnt}", file=stderr)
        if sample_cnt > expect_cnt:
            raise Exception("ratio implement error!")
        elif sample_cnt < expect_cnt:
            revise_idx = torch.cat((uniq_idx, torch.randperm(cnt, device=torch.device("cuda"))[:expect_cnt - sample_cnt]))
        else:
            revise_idx = uniq_idx
        if i != 0:
            revise_idx += new_offset[i - 1]
        idx.append(revise_idx)
    return torch.cat(idx)

import os
unique_idx = 0
def visualization(xyz, offset, layer, voxel_size, xtype="voxel_sample"):
    color_blue = np.array([0,170,255])
    path = "visual"
    # print(offset, file=stderr)
    if not os.path.exists(path):
        os.makedirs(path)
    global unique_idx
    for i in range(offset.size(0)):
        coord = xyz[offset[i - 1]:offset[i]] if i != 0 else xyz[:offset[i]]
        coord = coord.cpu().numpy()
        rgb = np.broadcast_to(color_blue, (coord.shape[0], color_blue.shape[0]))
        xyzrgb = np.concatenate((coord, rgb), axis=1)
        xyzrgb = [(xyzrgb[i, 0], xyzrgb[i, 1], xyzrgb[i, 2], xyzrgb[i, 3], xyzrgb[i, 4], xyzrgb[i, 5]) for i in range(xyzrgb.shape[0])]
        vertex = PlyElement.describe(np.array(xyzrgb, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        filepath = os.path.join(path, xtype +"_" + time.strftime("%Y-%m-%d_%H%M%S", time.localtime()) + "_layer" + str(layer) + "_voxel_size_" + str(voxel_size) + "_" + str(i) + "_" + str(unique_idx) + ".ply")
        PlyData([vertex]).write(filepath)
        print('PLY visualization file saved in', filepath, file=stderr)
        unique_idx += 1
        
def visual_union(xyz, offset, idx_vss, idx_fpss, layer):
    color_red = np.array([23,190,207]) # intersect
    color_green = np.array([152,223,138]) # vs
    color_blue = np.array([255,152,150]) # fps
    path = "visual_union/"
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(offset.size(0)):
        coord = xyz[offset[i - 1]:offset[i]] if i != 0 else xyz[:offset[i]]
        coord = coord.cpu().numpy()
        idx_vs = (idx_vss[offset[i - 1]:offset[i]] if i != 0 else idx_vss[:offset[i]]).cpu().numpy()
        idx_fps = (idx_fpss[offset[i - 1]:offset[i]] if i != 0 else idx_fpss[:offset[i]]).cpu().numpy()
        idx_intersect = np.intersect1d(idx_vs, idx_fps)
        idx_vs_unique = np.setdiff1d(idx_vs, idx_intersect)
        idx_fps_unique = np.setdiff1d(idx_fps, idx_intersect)
        coord_intersect = coord[idx_intersect, :]
        coord_vs_unique = coord[idx_vs_unique, :]
        coord_fps_unique = coord[idx_fps_unique, :]
        rgb_intersect = np.broadcast_to(color_blue, (coord_intersect.shape[0], color_blue.shape[0]))
        rgb_vs_unique = np.broadcast_to(color_green, (coord_vs_unique.shape[0], color_green.shape[0]))
        rgb_fps_unique = np.broadcast_to(color_red, (coord_fps_unique.shape[0], color_red.shape[0]))
        xyzrgb_intersect = np.concatenate((coord_intersect, rgb_intersect), axis=1)
        xyzrgb_vs_unique = np.concatenate((coord_vs_unique, rgb_vs_unique), axis=1)
        xyzrgb_fps_unique = np.concatenate((coord_fps_unique, rgb_fps_unique), axis=1)
        xyzrgb = np.concatenate((xyzrgb_intersect, xyzrgb_vs_unique, xyzrgb_fps_unique), axis=0)
        xyzrgb = [(xyzrgb[i, 0], xyzrgb[i, 1], xyzrgb[i, 2], xyzrgb[i, 3], xyzrgb[i, 4], xyzrgb[i, 5]) for i in range(xyzrgb.shape[0])]
        vertex = PlyElement.describe(np.array(xyzrgb, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        filepath = path + time.strftime("%Y-%m-%d_%H%M%S", time.localtime()) + "_layer" + str(layer) + "_" + str(i) + ".ply"
        PlyData([vertex]).write(filepath)
        print('PLY visualization file saved in', filepath, file=stderr)

def knn(x, k):
    """
    input x: (n, f)
    output idx: (n, k), dist: (m, k)
    """
    inner = -2*torch.matmul(x, x.transpose(0, 1))
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(0, 1)
    dist, idx = pairwise_distance.topk(k=k, dim=-1)   # (n, k)
    return idx, torch.sqrt(torch.abs(dist))

def knn_query(x, y, k):
    """
    input x: (n, f) y: (m, f)
    output idx: (n, k), dist: (n, k)
    """
    inner = -2 * torch.matmul(x, y.transpose(0, 1)) # (n, m)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) # (n, 1)
    yy = torch.sum(y ** 2, dim=1, keepdim=True) # (m, 1)
    pairwise_distance =  -xx - inner - yy.transpose(0, 1)
    # pairwise_distance = -torch.sum((x[:, None] - y[None]) ** 2, dim=-1) # (n, m)
    dist, idx = pairwise_distance.topk(k=k, dim=-1)   # (n, k)
    return idx, torch.sqrt(torch.abs(dist))

def knn_unaligned(nsample, xyz, new_xyz, offset, new_offset):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    output: idx: (m, nsample), dist2: (m, nsample)
    """
    if new_xyz is None: new_xyz = xyz
    assert xyz.is_contiguous() and new_xyz.is_contiguous()
    assert offset.size() == new_offset.size()
    idxs = []
    dists = []
    for i in range(new_offset.size(0)):
        if i == 0:
            new_coord = new_xyz[:new_offset[i], :]
            coord = xyz[:offset[i], :]
        else:
            new_coord = new_xyz[new_offset[i - 1]:new_offset[i], :]
            coord = xyz[offset[i - 1]:offset[i], :]
        # idx, dist = knn(new_coord, nsample)
        idx, dist = knn_query(new_coord, coord, nsample)
        idx += offset[i - 1] if i != 0 else 0
        idxs.append(idx)
        dists.append(dist)
    return torch.cat(idxs), torch.cat(dists)

def frnn_unaligned(nsample, xyz, new_xyz, offset, new_offset, radius=0.5):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    output: idx: (m, nsample), dist2: (m, nsample)
    """
    if new_xyz is None: new_xyz = xyz
    assert xyz.is_contiguous() and new_xyz.is_contiguous()
    assert offset.size() == new_offset.size()
    idxs = []
    dists = []
    for i in range(new_offset.size(0)):
        if i == 0:
            new_coord = new_xyz[:new_offset[i], :]
            coord = xyz[:offset[i], :]
        else:
            new_coord = new_xyz[new_offset[i - 1]:new_offset[i], :]
            coord = xyz[offset[i - 1]:offset[i], :]
        # idx, dist = knn_query(new_coord, coord, nsample)
        dist, idx, _, _ = frnn.frnn_grid_points(points1=new_coord.unsqueeze(0), points2=coord.unsqueeze(0), K=nsample, r=radius)  # (batch_size, num_points, k)
        dist, idx = dist.squeeze(0), idx.squeeze(0) # sqrt
        # a = torch.eq(idx, -1)
        # print(torch.nonzero(a).size(0))
        idx += offset[i - 1] if i != 0 else 0
        idxs.append(idx)
        dists.append(dist)
    return torch.cat(idxs), torch.cat(dists)

def get_max_min(xyz):
    xyz -= torch.min(xyz, dim=0)[0]
    xyz /= torch.max(xyz, dim=0)[0]

def frnn_query(nsample, xyz, new_xyz, offset, new_offset, radius=0.5):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    output: idx: (m, nsample), dist2: (m, nsample)
    """
    b, n_max, m_max = offset.size(0), offset[0], new_offset[0]
    f_num = xyz.size(1)
    query_xyz = []
    support_xyz = []
    query_length = [new_offset[0]]
    support_length = [offset[0]]
    for i in range(1, b):
        n_max = max(offset[i] - offset[i-1], n_max)
        m_max = max(new_offset[i] - new_offset[i-1], m_max)
        support_length.append(offset[i] - offset[i-1])
        query_length.append(new_offset[i] - new_offset[i-1])
    
    # normalize the point clouds
    # xyz_min = torch.min(xyz, dim=0)[0]
    # xyz_max = torch.max(xyz, dim=0)[0]
    # new_xyz_min = torch.min(new_xyz, dim=0)[0]
    # new_xyz_max = torch.max(new_xyz, dim=0)[0]

    # both_min = torch.min(xyz_min, new_xyz_min)
    # both_max = torch.max(xyz_max, new_xyz_max)

    # xyz -= both_min
    # xyz /= both_max
    # new_xyz -= both_min
    # new_xyz /= both_max
        
    support_length = torch.tensor(support_length).long().cuda()
    query_length = torch.tensor(query_length).long().cuda()
    coord = torch.cat((xyz[:offset[0], :], torch.zeros(n_max - offset[0], f_num).cuda()), dim=0)
    new_coord = torch.cat((new_xyz[:new_offset[0], :], torch.zeros(m_max - new_offset[0], f_num).cuda()), dim=0)
    support_xyz.append(coord)
    query_xyz.append(new_coord)

    for i in range(1, b):
        coord = torch.cat((xyz[offset[i-1]:offset[i], :], torch.zeros(n_max - (offset[i] - offset[i-1]), f_num).cuda()), dim=0)
        new_coord = torch.cat((new_xyz[new_offset[i-1]:new_offset[i], :], torch.zeros(m_max - (new_offset[i] - new_offset[i-1]), f_num).cuda()), dim=0)
        support_xyz.append(coord)
        query_xyz.append(new_coord)
    support_xyz = torch.stack(support_xyz, dim=0) # (b, n', 3)
    query_xyz = torch.stack(query_xyz, dim=0) # (b, m', 3)
    dist, idx, _, _ = frnn.frnn_grid_points(query_xyz, support_xyz, query_length, support_length, K=nsample, r=radius)
    
    # x = torch.cat([idx[i, :query_length[i], :] for i in range(b)], dim=0).cuda()
    # a = torch.eq(x, -1)
    # print(torch.nonzero(a).size(0))

    idx_squeezed = [idx[i, :query_length[i], :] + (offset[i-1] if i!=0 else 0) for i in range(b)]
    dist_squeezed = [dist[i, :query_length[i], :] for i in range(b)]
    idx = torch.cat(idx_squeezed, dim=0).cuda()
    dist = torch.cat(dist_squeezed, dim=0).cuda()

    return idx, dist

class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)

knnquery = KNNQuery.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None

grouping = Grouping.apply

import torch.autograd.profiler as profiler
def make_aligned(xyz, offset):
    # print(offset)
    # print(xyz.size())
    # assert(False)
    pre_offset = torch.cat((torch.tensor(0, dtype=torch.int32).to(torch.device("cuda")), offset[:-1]))
    max_point_num = torch.max(offset - pre_offset)
    last = 0
    aligned_xyz = []
    for i in offset:
        aligned_xyz.append(torch.cat((xyz[last:i, :], 2 * torch.ones((max_point_num - (i - last + 1), 3)).to(torch.device("cuda")))).unsqueeze(0)) # padding with point at infinity
        last = offset[i]
    return torch.cat(aligned_xyz, dim=0).contiguous()

def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True, radius=0.5, use_frnn=False):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    
    # use_event = True
    # if use_frnn:
    #     xyz = make_aligned(xyz, offset)
    #     if not (new_xyz is None):
    #         new_xyz = make_aligned(new_xyz, new_offset)
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        if glo.use_event:
            starter.record()
        # import numpy as np
        # np.savetxt("xyz.txt",xyz.cpu().numpy())
        # np.savetxt("new_xyz.txt", new_xyz.cpu().numpy())
        # np.savetxt("offset.txt", offset.cpu().numpy(), fmt="%d")
        # np.savetxt("new_offset.txt", new_offset.cpu().numpy(), fmt="%d")
        # print(f"nsample: {nsample}")
        if use_frnn:
            # idx, _ = frnn_unaligned(nsample, xyz, new_xyz, offset, new_offset, radius)
            idx, _ = frnn_query(nsample, xyz, new_xyz, offset, new_offset, radius)
        elif use_unaligned_knn:
            idx, _ = knn_unaligned(nsample, xyz, new_xyz, offset, new_offset)
        else:
            idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)
        if glo.use_event:
            ender.record()
            torch.cuda.synchronize()
            time_cost = starter.elapsed_time(ender)
            # time_cost = (end - start) * 1000
            glo.knn_time += time_cost
            # print(f"knn time cost: {time_cost}")

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)

    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat


class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape; nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        idx, = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(n, nsample, c, idx, grad_output, grad_input1, grad_input2)
        return grad_input1, grad_input2, None

subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert input.is_contiguous() and position.is_contiguous() and weight.is_contiguous()
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(n, nsample, c, w_c, input, position, weight, idx, output)
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight)
        return grad_input, grad_position, grad_weight, None

aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, radius, use_frnn, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if glo.use_event:
        starter.record()
    if use_frnn:
        # idx, dist = frnn_unaligned(k, xyz, new_xyz, offset, new_offset)
        idx, dist = frnn_query(k, xyz, new_xyz, offset, new_offset, radius)
    elif use_unaligned_knn:
        idx, dist = knn_unaligned(k, xyz, new_xyz, offset, new_offset)
    else:
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    if glo.use_event:
        ender.record()
        torch.cuda.synchronize()
        time_cost = starter.elapsed_time(ender)
        glo.knn_time += time_cost

    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        # starter.record()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, k), (n, k)
        # ender.record()
        # torch.cuda.synchronize()
        # global knn_time
        # knn_time += starter.elapsed_time(knn_time)
        dist_recip = 1.0 / (dist + 1e-8) # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        return None, None, grad_input, None, None, None

interpolation2 = Interpolation.apply
