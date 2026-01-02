from audioop import bias
import imp
from pickle import NONE
from turtle import width, window_width
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
import time
from utils.utils import project_function

class feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        net = []
        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # add net into class property
        self.features = nn.Sequential(*net)

    def forward(self, x):
        feature = self.features(x)
        return feature

def MLP(channels: list, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.SyncBatchNorm(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = [kpts.transpose(0, 1).reshape(1, 2, -1)]
        outputs = self.encoder(torch.cat(inputs, dim=1))
        return outputs


def attention(query, key, value):
    dim = query.shape[1]
    # scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    scores = torch.matmul(query.permute(0, 2, 3, 1), key.permute(0, 2, 1, 3)) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    # out = torch.einsum('bhnm,bdhm->bdhn', prob, value)
    out = torch.matmul(prob, value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return out, prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # x, _ = attention(query, key, value)
        dim = query.shape[1]
        # scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
        scores = torch.matmul(query.permute(0, 2, 3, 1), key.permute(0, 2, 1, 3)) / dim**.5
        prob = torch.nn.functional.softmax(scores, dim=-1)
        # out = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        x = torch.matmul(prob, value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))
        return out

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.Rope = RotaryEmbedding(256)

    def forward(self, x, source, tgt_pos, src_pos):
        message = self.attn(x + tgt_pos, source + src_pos, source)
        # q, k = self.Rope(x.permute(0, 2, 1), source.permute(0, 2, 1), 20, 15)
        # message = self.attn(q.permute(0, 2, 1), k.permute(0, 2, 1), source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, gather_homography):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.local_proj = MLP([800, 256, 256, 3])
        self.bias_proj = MLP([256, 256, 256, 6])
        self.pos_proj = MLP([8, 64, 128, 256])
        self.final_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True)
        self.pad = torch.nn.ZeroPad2d(2)
        self.kenc = KeypointEncoder(32, [8, 16, 32, 64])
        self.dropout1 = torch.nn.Dropout(0.0)
        self.dropout2 = torch.nn.Dropout(0.0)
        self.dim = feature_dim
        self.gather_homography = gather_homography

    def estimate_medium_information(self, desc0, desc1, h, w):
        cols = torch.arange(-2, 3, device=desc0.device).reshape(5, 1).repeat(1, 5).reshape(-1)
        rows = torch.arange(-2, 3, device=desc0.device).reshape(1, 5).repeat(5, 1).reshape(-1)
        kpts = torch.zeros((25), 2, device=desc0.device)
        kpts[:, 0] = cols
        kpts[:, 1] = rows
        self.kpts_local = self.kenc(kpts)
        scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        scores = scores / self.dim ** .5
        b, m, n = scores.shape
        scores = scores.softmax(dim=2)
        indices_medium = torch.argmax(scores, dim=2)
        i = indices_medium // w
        j = indices_medium % w
        index_extract = torch.stack([i, j], -1)[:, :, None, None, :].repeat(1, 1, 5, 5, 1)
        desc1 = self.pad(desc1.reshape(b, self.dim, h, w)).reshape(b, self.dim, -1)
        grid_x, grid_y = torch.meshgrid(torch.arange(5, device=desc0.device), torch.arange(5, device=desc0.device))
        index_extract = torch.stack([grid_x, grid_y],dim=-1)[None, None].repeat(b, m, 1, 1, 1) + index_extract
        index_extract = index_extract[..., 0] * (w + 2) + index_extract[..., 1]
        feature_extract = torch.gather(desc1, 2, index_extract.reshape(b, 1, -1).expand(-1, self.dim, -1)).\
            reshape(b, self.dim // 8, 8, m, 5, 5).permute(0, 1, 4, 5, 3, 2).reshape(b, self.dim // 8, -1, m, 8)
        feature_extract = torch.einsum('bftmi,bfmi->bftm', feature_extract, desc0.reshape(b, self.dim // 8, 8, m).permute(0, 
            1, 3, 2))
        feature_extract = (self.kpts_local[:, :, :, None] + feature_extract).reshape(b, -1, m)
        p = self.local_proj(feature_extract)
        p_new = self.dropout1(p).sigmoid()
        bias = self.bias_proj(desc0)
        # bias的八个项分别是：y坐标，x坐标，尺度，旋转，四个偏置
        y = i.float() + 0.5 + (p_new[:, 0, :] - 0.5) * 4
        x = j.float() + 0.5 + (p_new[:, 1, :] - 0.5) * 4
        bias_update = torch.cat([inverse_sigmoid(y[:, None] / h), inverse_sigmoid(x[:, None] / w), bias, p[:, 2, None]], dim=1)
        bias = self.dropout2(bias).sigmoid()
        confidence = p_new[:, 2, :]
        scale = torch.exp(bias[:, 0, :] * math.log(64.0) - math.log(64.0) / 2)
        rotation = (bias[:, 1, :] - 0.5) * math.pi * 2
        normal = (bias[:, 2:, :].permute(0, 2, 1) - 0.5)
        grid_x, grid_y = torch.meshgrid(torch.arange(h, device=y.device) + 0.5, torch.arange(w, device=y.device) + 0.5)
        grid = torch.stack([grid_y, grid_x], -1).reshape(-1, 2)[None].expand(y.shape[0], -1, -1).repeat(1, 1, 4)
        H_matrix = construct_homography(y, x, scale, rotation, normal, grid, w, h, self.gather_homography)
        # pos_embed = self.pos_proj(torch.cat([y[:, None] / h, x[:, None] / w, scale[:, None], rotation[:, None], normal.permute(0, 2, 1)], dim=1))
        # return {"indices_medium": indices_medium, "y": y, "x": x, "scale": scale, "rotation":rotation,
        #     "normal": normal, "scores": scores, "confidence": confidence}
        return {"indices_medium": indices_medium, "H_matrix": H_matrix, "scores": scores, "confidence": confidence, "p": p_new}, bias_update

    def forward(self, desc0, desc1, h, w, pos_embed):
        results = []
        num = 0
        # pos_embed2 = pos_embed
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
                delta0, delta1 = layer(desc0, src0, pos_embed, pos_embed), layer(desc1, src1, pos_embed, pos_embed)
                desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
                # delta0 = layer(desc0, src0, pos_embed, pos_embed)
                # desc0 = (desc0 + delta0)
                if num == 9:
                    mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
                    result, bias_update = self.estimate_medium_information(mdesc0, mdesc1, h, w)
                    # pos_embed2 = result["pos_embed"]
                    results.append(result)
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                delta0, delta1 = layer(desc0, src0, pos_embed, pos_embed), layer(desc1, src1, pos_embed, pos_embed)
                desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            num += 1
        return desc0, desc1, results, bias_update

def construct_homography(y, x, scale, rotation, normal, grid, width, height, gather_homography=None, patch_size_high=32, search_distance=1):
    # output: (b, h, w, proposals, 3, 3)
    search_range = 2 * search_distance + 1
    H_matrix = torch.zeros([y.shape[0], y.shape[1], 9], device=x.device)
    # grid_x, grid_y = torch.meshgrid(torch.arange(height, device=y.device) + 0.5, torch.arange(width, device=y.device) + 0.5)
    # grid = torch.stack([grid_y, grid_x], -1).reshape(-1, 2)[None].expand(y.shape[0], -1, -1).repeat(1, 1, 4)
    basic_change = torch.tensor([-1, -1, 1, -1, -1, 1, 1, 1], device=scale.device).reshape(1, 1, 8).expand(scale.shape[0], scale.shape[1], -1)\
        * patch_size_high // 2
    source_points = (grid * patch_size_high + basic_change).reshape(scale.shape[0], scale.shape[1], 4, 2)
    scales_use = scale[:, :, None].expand(-1, -1, 8)
    cos_angle = torch.cos(rotation[:, :, None])
    sin_angle = torch.sin(rotation[:, :, None])
    basic_change = basic_change + torch.cat([-normal[:, :, 0, None] - normal[:, :, 1, None], -normal[:, :, 2, None] - normal[:, :, 3, None],
        normal[:, :, 0, None] - normal[:, :, 1, None], normal[:, :, 2, None] - normal[:, :, 3, None],
        -normal[:, :, 0, None] + normal[:, :, 1, None], -normal[:, :, 2, None] + normal[:, :, 3, None],
        normal[:, :, 0, None] + normal[:, :, 1, None], normal[:, :, 2, None] + normal[:, :, 3, None],], dim=2)
    basic_change2 = torch.cat([basic_change[:, :, 0, None] * cos_angle - basic_change[:, :, 1, None] * sin_angle, 
        basic_change[:, :, 1, None] * cos_angle + basic_change[:, :, 0, None] * sin_angle,
        basic_change[:, :, 2, None] * cos_angle - basic_change[:, :, 3, None] * sin_angle, 
        basic_change[:, :, 3, None] * cos_angle + basic_change[:, :, 2, None] * sin_angle,
        basic_change[:, :, 4, None] * cos_angle - basic_change[:, :, 5, None] * sin_angle, 
        basic_change[:, :, 5, None] * cos_angle + basic_change[:, :, 4, None] * sin_angle,
        basic_change[:, :, 6, None] * cos_angle - basic_change[:, :, 7, None] * sin_angle, 
        basic_change[:, :, 7, None] * cos_angle + basic_change[:, :, 6, None] * sin_angle], dim=2) 
    # print(bias_use.sum())
    target_points = (torch.stack([x, y], -1).repeat(1, 1, 4) * patch_size_high + basic_change2 * scales_use).reshape(scale.shape[0], scale.shape[1], 4, 2)
    a_mat = torch.zeros([scale.shape[0], scale.shape[1], 4, 2, 8], device=scale.device).float()
    a_mat[:, :, :, 0, :2] = source_points
    a_mat[:, :, :, 0, 2] = 1
    a_mat[:, :, :, 0, 6:8] = - source_points * target_points[:, :, :, 0, None].expand(-1, -1, -1, 2)
    a_mat[:, :, :, 1, 3:5] = source_points
    a_mat[:, :, :, 1, 5] = 1
    a_mat[:, :, :, 1, 6:8] = - source_points * target_points[:, :, :, 1, None].expand(-1, -1, -1, 2)
    a_mat = a_mat.reshape(a_mat.shape[0], a_mat.shape[1], 8, 8)
    p_mat = target_points.reshape(a_mat.shape[0], a_mat.shape[1], 8)
    H_matrix[:, :, :8] = torch.einsum('bshp,bsp->bsh', torch.linalg.inv(a_mat), p_mat)
    H_matrix[:, :, 8] = 1.0
    if gather_homography==None:
        return H_matrix.reshape(H_matrix.shape[0], 3, 3)
    else:
        H_matrix = gather_homography(H_matrix.reshape(y.shape[0], height, width, 9).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).\
            reshape(y.shape[0], height, width, search_range * search_range, 3, 3)
        # test_result = torch.einsum('bhwsij,tj->bhwsti', H_matrix, torch.cat([grid[0, :, :2] * patch_size_high, torch.ones_like(grid[0, :, 0, None])], dim=1))
        return H_matrix




def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, one, ns, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    # one = scores.new_tensor(1)
    # ms, ns = (m*one), (n*one)
    ms = (m*one)
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)
    norm = - (ms + ns.sum(2).reshape(-1)).reshape(-1, 1).log()
    # print(ms + ns.sum(2).reshape(-1))
    log_nu = torch.cat([ns.log().reshape(b, -1) + norm, ms.log()[None] + norm], 1)
    log_mu = torch.cat([norm.expand(-1, m), ns.sum(2).log().reshape(-1, 1) + norm], 1)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = (Z.permute(1, 2, 0) - norm.reshape(-1)).permute(2, 0, 1)  # multiply probabilities by M+N
    return Z


def log_optimal_transport2(scores, one, ns, iters: int):
    b, m, n = scores.shape
    # one = scores.new_tensor(1)
    # ms, ns = (m*one), (n*one)
    ms = ((m - 1)*one)
    # bins0 = alpha.expand(b, m, 1)
    # bins1 = alpha.expand(b, 1, n)
    # alpha = alpha.expand(b, 1, 1)
    couplings = scores
    # couplings = torch.cat([torch.cat([scores, bins0], -1),
    #                        torch.cat([bins1, alpha], -1)], 1)
    norm = - (ms + ns.sum(2).reshape(-1)).reshape(-1, 1).log()
    # print(ms + ns.sum(2).reshape(-1))
    log_nu = torch.cat([ns.log().reshape(b, -1) + norm, ms.log()[None] + norm], 1)
    log_mu = torch.cat([norm.expand(-1, m - 1), ns.sum(2).log().reshape(-1, 1) + norm], 1)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = (Z.permute(1, 2, 0) - norm.reshape(-1)).permute(2, 0, 1)  # multiply probabilities by M+N
    return Z

# patch内部的scores的输入是（b * m * 2）, alpha是（b * 2）
# batch内部的scores的输入是（1 * b * 3）， alpha是（1 * 3） 
def log_optimal_transport_eval(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    couplings = scores
    ns = alpha
    norm = - alpha.sum(1).reshape(-1, 1).log()
    log_nu = ns.log().reshape(b, -1) + norm
    log_mu = norm.expand(-1, m)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = (Z.permute(1, 2, 0) - norm.reshape(-1)).permute(2, 0, 1)  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(128, 128)):
        super().__init__()
        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

def inverse_sigmoid(x, eps=1e-8):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

# 将对应位置的特征抽出，为deformable transformer做准备, positions的维度是N,1,W,2
def feature_sample(feature, positions, num_heads=1):
    # N, D, H, W =feature.shape
    output = F.grid_sample(feature, positions, mode='bilinear', padding_mode='zeros', align_corners=False)
    return output.reshape(feature.shape[0], feature.shape[1], -1)


def runtime_sample_positions(ref_positions, H_matrix, height, width, radius, patch_scale=32):
    # H_matrix: (b2,3,3)
    # ref_feature/tgt_feature: (b2,c,h,w)
    # ref_positions: (b2,2)，这里输入的是现实坐标还是归一化坐标呢？先设定成现实坐标吧
    delta = torch.cat([torch.tensor([[-1, -1]]), torch.tensor([[-1, 0]]), torch.tensor([[-1, 1]]), torch.tensor([[0, -1]]), torch.tensor([[0, 0]]),
        torch.tensor([[0, 1]]), torch.tensor([[1, -1]]), torch.tensor([[1, 0]]), torch.tensor([[1, 1]])], dim=0).to(H_matrix.device)
    ref_positions_use = ref_positions
    ref_positions_use = ref_positions_use[:, None] + delta[None] * patch_scale // 4 * radius.reshape(-1, 1, 1)
    ref_positions_use = torch.cat([ref_positions_use, torch.ones([ref_positions_use.shape[0], 
        ref_positions_use.shape[1], 1], device=H_matrix.device)], dim=-1)
    tgt_positions = torch.matmul(H_matrix[:, None].expand(-1, 9, -1, -1).detach(), ref_positions_use[..., None].detach()).squeeze(3)
    tgt_positions = tgt_positions / (tgt_positions[:, :, 2, None] + 1e-9)
    tgt_positions[:, :, 1] = tgt_positions[:, :, 1] / (height * patch_scale // 2) - 1
    tgt_positions[:, :, 0] = tgt_positions[:, :, 0] / (width * patch_scale // 2) - 1
    ref_positions_use[:, :, 1] = ref_positions_use[:, :, 1] / (height * patch_scale // 2) - 1
    ref_positions_use[:, :, 0] = ref_positions_use[:, :, 0] / (width * patch_scale // 2) - 1
    return ref_positions_use[:, :, :2], tgt_positions[:, :, :2]

def runtime_feature_sample(ref_feature, tgt_feature, ref_positions, tgt_positions):
    # positions: (b2, 9 ,3)
    # feature(b, c, h, w)
    ref_feature_new = feature_sample(ref_feature[None].transpose(1, 2), ref_positions.reshape(1, 1, 1, -1, 
        3)).reshape(-1, ref_positions.shape[0], 9).permute(1, 0, 2)
    tgt_feature_new = feature_sample(tgt_feature[None].transpose(1, 2), tgt_positions.reshape(1, 1, 1, -1, 
        3)).reshape(-1, ref_positions.shape[0], 9).permute(1, 0, 2)
    return ref_feature_new, tgt_feature_new


def duplicate_interleave(m):
    return m.view(-1, 1).repeat(1, 2).view(m.shape[0], -1)


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int = 10_000,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim / 2.0)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim // 2
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def cos_sin(
        self,
        seq_len: int,
        device: str = "cpu",
        dtype=torch.bfloat16,
    ):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
            emb = duplicate_interleave(freqs).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, ...].type(dtype)
            self.sin_cached = emb.sin()[None, ...].type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k, h, w):
        """
        Designed to operate on queries and keys that are compatible with
        [batch_size, n_heads_per_partition, seq_len, head_dim]
        """
        batch, seq_len, head_dim = q.shape
        cos_h, sin_h = self.cos_sin(h, q.device, q.dtype)
        cos_h = cos_h[:, :, None].repeat(1, 1, w, 1).reshape(1, seq_len, -1)
        sin_h = sin_h[:, :, None].repeat(1, 1, w, 1).reshape(1, seq_len, -1)
        cos_w, sin_w = self.cos_sin(w, q.device, q.dtype)
        cos_w = cos_w.repeat(1, h, 1).reshape(1, seq_len, -1)
        sin_w = sin_w.repeat(1, h, 1).reshape(1, seq_len, -1)
        q_new = torch.cat([(q[:, :, :head_dim // 2] * cos_h) + (rotate_every_two(q[:, :, :head_dim//2]) * sin_h), 
            (q[:, :, head_dim // 2:] * cos_w) + (rotate_every_two(q[:, :, head_dim//2:]) * sin_w)], -1)
        k_new = torch.cat([(k[:, :, :head_dim // 2] * cos_h) + (rotate_every_two(k[:, :, :head_dim//2]) * sin_h), 
            (k[:, :, head_dim // 2:] * cos_w) + (rotate_every_two(k[:, :, head_dim//2:]) * sin_w)], -1)        
        return q_new, k_new

    def forward_one(self, q, h, w):
        """
        Designed to operate on queries and keys that are compatible with
        [batch_size, n_heads_per_partition, seq_len, head_dim]
        """
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        return (q * cos) + (rotate_every_two(q) * sin)