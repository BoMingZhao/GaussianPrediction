import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, output_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        output_dim = default(output_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = mask>0
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

def positional_encoding(positions, freqs, ori=False):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1] + (pts.shape[-1] * 2 + ori_c, ))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1] + (pts.shape[-1] * 2,))
    return pts

class Deformable_Field(nn.Module):
    def __init__(self, input_dim, output_dim=10, d=8, w=256, use_softmax=False, split_xyz=False):
        super(Deformable_Field, self).__init__()
        self.input_dim = input_dim
        self.d = d
        self.w = w
        self.use_softmax = use_softmax
        self.output_dim = output_dim

        self.split_xyz = split_xyz
        if self.split_xyz:
            self.output_times = self.output_dim
            self.output_dim = 1
            self.mlp = {}
            self.feature_to_deformation = {}
            for times in range(self.output_times):
                mlp = []
                for i in range(d):
                    if i == 0:
                        mlp.append(nn.Linear(self.input_dim, self.w))
                    else:
                        mlp.append(nn.Linear(self.w, self.w))
                    mlp.append(nn.ReLU())
                self.mlp[f"mlp{times:d}"] = nn.Sequential(*mlp)
                self.feature_to_deformation[f"feature_to_deformation{times:d}"] = nn.Sequential(nn.Linear(w, self.output_dim))
            self.mlp = nn.ModuleDict(self.mlp)
            self.feature_to_deformation = nn.ModuleDict(self.feature_to_deformation)
        else:
            mlp = []
            for i in range(d):
                if i == 0:
                    mlp.append(nn.Linear(self.input_dim, self.w))
                else:
                    mlp.append(nn.Linear(self.w, self.w))
                mlp.append(nn.ReLU())
            self.mlp = nn.Sequential(*mlp)
            self.feature_to_deformation = nn.Sequential(nn.Linear(w, self.output_dim))

    def forward(self, x):
        if self.split_xyz:
            raw_list = []
            for i in range(self.output_times):
                feature = self.mlp[f"mlp{i:d}"](x)
                raw_output = self.feature_to_deformation[f"feature_to_deformation{i:d}"](feature)
                if self.use_softmax:
                    raw_output = nn.Softmax(dim=-1)(raw_output)
                raw_list += [raw_output]
            raw_output = torch.cat(raw_list, dim=-1)
        else:
            feature = self.mlp(x)
            raw_output = self.feature_to_deformation(feature)
            if self.use_softmax:
                raw_output = nn.Softmax(dim=-1)(raw_output)
        return raw_output
    
class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=False, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.InstanceNorm1d(512)
        self.bn2 = nn.InstanceNorm1d(256)
        self.bn3 = nn.InstanceNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x
        # return x, trans_feat


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss