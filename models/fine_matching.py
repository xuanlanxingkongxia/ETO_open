import math
import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, 1, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f1.shape
        W = int(math.sqrt(WW))
        scale = 2
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale
        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return
        feat_f0_picked = feat_f0.squeeze(1)
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        local_max = torch.argmax(sim_matrix, 1)
        softmax_temp = 1. / C**.5
        sim_matrix = softmax_temp * sim_matrix
        data.update({'refine_scores': torch.softmax(sim_matrix.clone(), dim=1), 'refine_choice': local_max})
        # cols = torch.arange(7).reshape(7, 1).repeat(1, 7).reshape(-1)
        # rows = torch.arange(7).reshape(1, 7).repeat(7, 1).reshape(-1)
        # kpts = torch.zeros((49), 2).to(feat_f0.device)
        # kpts[:, 0] = rows
        # kpts[:, 1] = cols
        # kpts = kpts[None]
        # sim_matrix[torch.logical_or((local_max[:, None] % 7 - kpts[:, :, 0]).abs() > 1.5, (local_max[:, None] // 7 - kpts[:, :, 1]).abs() > 1.5)] = -1e7 
        heatmap = torch.softmax(sim_matrix, dim=1).view(-1, W, W)
        # print(heatmap.reshape(-1, W * W).max(1)[0].median())
        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        # grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        # var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        # std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
        
        # for fine-level supervision
        # data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })