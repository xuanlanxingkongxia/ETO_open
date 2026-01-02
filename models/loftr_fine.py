from re import A
from numpy.lib.function_base import average, select
import torch
import torch.nn as nn
from einops.einops import rearrange
import torchvision

from my_third_party.LoFTR.src.loftr.backbone import build_backbone
from my_third_party.LoFTR.src.loftr.utils.position_encoding import PositionEncodingSine
from my_third_party.LoFTR.src.loftr.loftr_module import LocalFeatureTransformer, FinePreprocess
from my_third_party.LoFTR.src.loftr.utils.fine_matching import FineMatching
import torch.nn.functional as F
from models.modules import *
from kornia.utils.grid import create_meshgrid
import torchvision.models as models

from utils.utils import Position_loss, origin_extract
from resnet import ResNet2, BasicBlock


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# def conv3x3_add_padding(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=2, bias=False)

class FPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """
    def __init__(self):
        super().__init__()
        block_dims = [128, 192, 264]
        block_dims_before = [64, 64, 128]
        self.layer3_outconv = conv1x1(block_dims_before[2], block_dims[2])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[2]),
        )
        self.layer2_outconv = conv1x1(block_dims_before[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims_before[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        self.pad2 = torch.nn.ZeroPad2d(2)
        self.pad1 = torch.nn.ZeroPad2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, x_before):
        # ResNet Backbone
        x3_out = self.layer3_outconv2(x) + self.layer3_outconv(x_before[2])

        x3_out_2x = self.pad1(F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=False))
        x2_out = self.pad1(self.layer2_outconv(x_before[1]))
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=False)
        x1_out = self.pad2(self.layer1_outconv(x_before[0]))
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # x1_out = F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=False)
        # x1_out = F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True)
        x_out = x1_out.reshape(2, -1, 128, x1_out.shape[2], x1_out.shape[3])
        return x_out[0], x_out[1]


class LoFTR_fine(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        # Modules
        self.backbone = FPN_8_2()
        # self.loftr_fine = LocalFeatureTransformer(config["fine"])
        # self.fine_matching = FineMatching(if_new=True)
        self.scale_proj = nn.Conv2d(in_channels=128, out_channels=1,
                                     kernel_size=3, padding=1, stride=1, bias=True)
        self.compress = MLP([264, 264, 264, 128])
        # self.gnn = AttentionalGNN(128, ['self', 'cross'] * 5)
        # self.final_proj = nn.Conv1d(128, 128, kernel_size=1, bias=True)
        self.pad = torch.nn.ZeroPad2d(2)
        self.pad_1 = torch.nn.ConstantPad2d(2, 1e-2)
        self.sigmoid = nn.Sigmoid()
        self.kenc = KeypointEncoder(
            128, [32, 64, 128, 256, 512])
        self.descriptor_extract = ResNet2(BasicBlock, [3, 4, 6, 3])
        pretrained_dict = models.resnet34(pretrained=True).state_dict()
        model_dict = self.descriptor_extract.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.descriptor_extract.load_state_dict(model_dict)
        for p in self.descriptor_extract.parameters():
           p.requires_grad = True


    def forward(self, data, mdesc, desc_before):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        pic0 = torch.cat([data["new_left"].permute(0, 3, 1, 2).float().contiguous(), data["new_right"].permute(0, 3, 1, 2).float().contiguous()], dim=0).reshape(-1, 
            data["new_left"].shape[3], data["new_left"].shape[1], data["new_left"].shape[2])
        desc_before = self.descriptor_extract.forward2(pic0)
        self.one = torch.tensor(1.0, device=mdesc.device)
        feat_f0, feat_f1 = self.backbone(mdesc[:, :, :-1].reshape(mdesc.shape[0], -1, 12, 12), desc_before)
        # 计算垃圾箱的特征
        rubbish = self.compress(mdesc[:, :, :-1].reshape(mdesc.shape[0], -1, 144))
        W = 8
        # 额外进行了两次padding，所以特征图的大小是52*52而不是48*48，所以所有坐标要+2
        M = 52
        T = 5
        # 计算上一层传入的点所在的 W * W patch位置，包括rubbish
        b = data['b_ids'].reshape(-1, 1).repeat(1, W*W)
        data['mkpts0_c'] = torch.round(data['mkpts0_c'] / 4.0).long() * 4
        x0 = (data['mkpts0_c'][:, 0] // 2).reshape(-1, 1).expand(-1, W * W) + torch.arange(W, device=feat_f0.device).reshape(1, 1, W).repeat(data['b_ids'].shape[0], W, 1).reshape(-1, W*W) - W / 2 + 2
        y0 = (data['mkpts0_c'][:, 1] // 2).reshape(-1, 1).expand(-1, W * W) + torch.arange(W, device=feat_f0.device).reshape(1, W, 1).repeat(data['b_ids'].shape[0], 1, W).reshape(-1, W*W) - W / 2 + 2
        index0 = (b * M * M + y0 * M + x0).long().reshape(-1, 1).expand(-1, 128)
        data['mkpts1_c'] = torch.where(data['mkpts1_c'] >= 96, torch.tensor(96 , device=feat_f0.device).double(), data['mkpts1_c'])
        data['mkpts1_c'] = torch.where(data['mkpts1_c'] <= 0, torch.tensor(0, device=feat_f0.device).double(), data['mkpts1_c'])
        data['mkpts1_c'] = torch.round(data['mkpts1_c'] / 4.0).long() * 4
        x1 = (data['mkpts1_c'][:, 0] // 2).reshape(-1, 1).expand(-1, W*W) + torch.arange(W, device=feat_f0.device).reshape(1, 1, W).repeat(data['b_ids'].shape[0], W, 1).reshape(-1, W*W) - W / 2 + 2
        y1 = (data['mkpts1_c'][:, 1] // 2).reshape(-1, 1).expand(-1, W*W) + torch.arange(W, device=feat_f0.device).reshape(1, W, 1).repeat(data['b_ids'].shape[0], 1, W).reshape(-1, W*W) - W / 2 + 2
        index1 = (b * M * M + y1 * M + x1).long().reshape(-1, 1).expand(-1, 128)

        cols = torch.arange(0, W).reshape(W, 1).repeat(1, W).reshape(-1) / float(W)
        rows = torch.arange(0, W).reshape(1, W).repeat(W, 1).reshape(-1) / float(W)
        kpts  = torch.zeros((W * W), 2).to(feat_f0.device)
        kpts [:, 0] = cols
        kpts [:, 1] = rows

        feat_f0_unfold = torch.gather(feat_f0.permute(0, 2, 3, 1).reshape(-1, feat_f0.shape[1]), 0, index0).reshape(-1, W*W, 128).permute(0, 2, 1) + self.kenc(kpts)
        feat_f1_unfold = torch.gather(feat_f1.permute(0, 2, 3, 1).reshape(-1, feat_f1.shape[1]), 0, index1).reshape(-1, W*W, 128).permute(0, 2, 1) + self.kenc(kpts)
        x2 = torch.round(data['mkpts0_c'][:, 0] / 8.0).long()
        y2 = torch.round(data['mkpts0_c'][:, 1] / 8.0).long()
        index2 = (data['b_ids'] * 12 * 12 + y2 * 12 + x2).long().reshape(-1, 1).expand(-1, 128)
        rubbish_unfold = torch.gather(rubbish.permute(0, 2, 1).reshape(-1, 128), 0, index2).reshape(-1, 128, 1)
        feat_f0_unfold = torch.cat([feat_f0_unfold, rubbish_unfold], dim=2)
        feat_f1_unfold = torch.cat([feat_f1_unfold, rubbish_unfold], dim=2)
        # self 与 cross attention
        feat_f0_unfold, feat_f1_unfold = self.gnn(feat_f0_unfold, feat_f1_unfold)
        # feat_f0_unfold, feat_f1_unfold = self.final_proj(feat_f0_unfold), self.final_proj(feat_f1_unfold)
        # 估计局部scale
        scale = self.scale_proj(feat_f1_unfold[:, :, :-1].reshape(-1, 128, W, W)).reshape(-1, 1, W*W)
        scale = torch.exp(self.sigmoid(scale) * math.log(256.0) - math.log(256.0) / 2)
        scale_x = (scale + 1e-8).sqrt() 
        scale_y = (scale + 1e-8).sqrt() 
        # 计算特征的相关性分数
        scores = torch.einsum('bdn,bdm->bnm', feat_f0_unfold, feat_f1_unfold)
        scores = scores / 128**.5
        scores_origin = log_optimal_transport2(0.1 * scores, self.one, scale, iters=100)
        scores = torch.exp(scores_origin)
        # # 考虑到5*5的邻域可能会超越边界，所以事先在周围填充了一层0
        # scores_back = self.pad(scores[:, :-1, :-1].reshape(scores.shape[0], scores.shape[1] - 1, W, W)).reshape(scores.shape[0], scores.shape[1] - 1, -1)
        # # 只需要计算左图中间的4*4的匹配即可
        # scores_back = scores_back.reshape(scores.shape[0], W, W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1)
        # # 提取T*T邻域， 并计算回归坐标结果
        # max0 = scores[:, :-1, :-1].max(2)[1].reshape(scores.shape[0], W, W)[:, 2:6, 2:6].reshape(scores.shape[0], 16)
        # x3 = (max0 % W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=feat_f0.device).reshape(1, 1, 1, T).repeat(max0.shape[0], max0.shape[1], T, 1).reshape(max0.shape[0], max0.shape[1], T**2)
        # y3 = (max0 // W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=feat_f0.device).reshape(1, 1, T, 1).repeat(max0.shape[0], max0.shape[1], 1, T).reshape(max0.shape[0], max0.shape[1], T**2)
        # index3 = y3 * (W + 4) + x3
        # # print(index3.max(), index3.min())
        # # import pdb
        # # pdb.set_trace()
        # scale_x_new = self.pad_1(scale_x.reshape(1, -1, W, W)).reshape(scale_x.shape[0], 1, -1).expand(-1, 16, -1)
        # scale_y_new = self.pad_1(scale_y.reshape(1, -1, W, W)).reshape(scale_y.shape[0], 1, -1).expand(-1, 16, -1)
        # scores_unfold_x = torch.gather((scores_back + 1e-7).sqrt() / scale_x_new, 2, index3)
        # scores_unfold_y = torch.gather((scores_back + 1e-7).sqrt() / scale_y_new, 2, index3)
        # positions = create_meshgrid(T, T, False, device=feat_f0.device).reshape(-1, 2) * 2 - (T - 1)
        # weighted_point_x = torch.einsum('ijp,p->ij', scores_unfold_x, positions[:, 0])
        # weighted_point_y = torch.einsum('ijp,p->ij', scores_unfold_y, positions[:, 1])
        # point_weight_sum_x = scores_unfold_x.sum(2)
        # point_weight_sum_y = scores_unfold_y.sum(2)
        # # 计算真实坐标系下的坐标点
        # mkpts1_f = torch.zeros([scale.shape[0], 16, 2], device=scores.device)
        # mkpts1_f[:, :, 0] = weighted_point_x / point_weight_sum_x + (max0 % W + 0.5 - W/2) * 2.0
        # mkpts1_f[:, :, 1] = weighted_point_y / point_weight_sum_y + (max0 // W + 0.5 - W/2) * 2.0
        # mkpts1_f = mkpts1_f + data['mkpts1_c'].reshape(-1, 1, 2).repeat(1, 16, 1)
        # mkpts0_f = data['mkpts0_c'].reshape(-1, 1, 2).repeat(1, 16, 1) + create_meshgrid(4, 4, False, device=feat_f0.device)\
        #     .reshape(1, -1, 2).repeat(data['mkpts0_c'].shape[0], 1, 1) * 2 - 3.0
        # # 计算whole_loss
        # scores_unfold_sum = torch.gather(scores_back, 2, index3).sum(2)
        # whole_loss = (torch.ones([scores_back.shape[0], scores_back.shape[1]], device=feat_f0.device) - scores_unfold_sum)
        # whole_loss = torch.where(whole_loss >= 1e-2, whole_loss, torch.tensor(0.0, device=mdesc.device)) / ((whole_loss >= 1e-2).float().sum() + 10) / 10
        mkpts0_f, mkpts1_f, whole_loss = self.Compute_result(scores, W, T, scale_x, scale_y, data['mkpts0_c'], data['mkpts1_c'], feat_f0_unfold.device)

        mkpts1_f_reverse, mkpts0_f_reverse, whole_loss_reverse = self.Compute_result_reverse(scores.permute(0, 2, 1), W, T, 
            torch.ones([scores.shape[0], 1, scores.shape[1] - 1], device=feat_f0.device), torch.ones([scores.shape[0], 1, scores.shape[1] - 1], device=feat_f0.device),
             data['mkpts1_c'], data['mkpts0_c'], feat_f0_unfold.device)
        
        label = ((data['label_dense'][:, :, :2] - data['mkpts1_c'][:, None, :].expand(-1, 16, -1)) + W + 1e8).reshape(-1, 2)
        # criterion1 = torch.logical_and(label[:, 0] >= 0, label[:, 0] < 16)
        # criterion2 = torch.logical_and(label[:, 1] >= 0, label[:, 1] < 16)
        # criterion = torch.logical_and(criterion1, criterion2)
        # label[:, 0] = torch.where(torch.logical_and(data['label_dense'][:, :, 0].reshape(-1) > -10.01, data['label_dense'][:, :, 0].reshape(-1) < -9.99), torch.tensor(-10.0, device=label.device).double(), label[:, 0])
        # label[:, 0] = torch.where(criterion, label[:, 0], torch.tensor(-100000.0, device=label.device).double())

        # criterion1 = torch.logical_and(label[:, 0] >= 1, label[:, 0] < 15)
        # criterion2 = torch.logical_and(label[:, 1] >= 1, label[:, 1] < 15)
        # criterion = torch.logical_and(criterion1, criterion2)
        # label[:, 0] = torch.where(criterion, label[:, 0], torch.tensor(-10.0, device=label.device).double())

        # select = torch.arange(label.shape[0], device=scores.device) % 16 == 10
        # select1 = torch.logical_or(torch.arange(label.shape[0], device=scores.device) % 16 == 5, torch.arange(label.shape[0], device=scores.device) % 16 == 15)
        # select2 = torch.logical_or(torch.arange(label.shape[0], device=scores.device) % 16 == 7, torch.arange(label.shape[0], device=scores.device) % 16 == 13)
        # select = torch.logical_or(select1, select2)
        # select = torch.logical_or(select2, select)
        # label[:, 0] = torch.where(select, label[:, 0], torch.tensor(-10.0, device=label.device).double())

        
        scores_used = scores[:, :-1, :].reshape(scores.shape[0], W, W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1) + 1e-8
        position_loss, average_error, _ = Position_loss(label.reshape(-1, 16, 2), ((mkpts1_f - data['mkpts1_c'].unsqueeze(1).expand(-1, 16, -1)) / 2 + 4)[:, :, [1, 0]], 
            scores_used, scores_used[:, :, :-1].max(2)[1], 2, 8, 8, weight=8)
        position_loss = position_loss

        # scores_used[:, :, -1] *= 2
        if_matching1 = (scores_used.max(2)[1] != W ** 2)
        if_matching2 = (scores[:, :, :-1].max(1)[1] != W ** 2)
        nomatching_criterion1 = torch.logical_and(if_matching1, label[:, 0].reshape(-1, 16) < -100.0)
        # print(scores[:, -1, :-1].mean(0).sum(), scores_used[:, :, -1].mean(0).sum())

        # label[:, 0] = torch.where(if_matching1.reshape(-1), label[:, 0], torch.tensor(-10.0, device=label.device).double())


        nomatching_loss = - torch.where(nomatching_criterion1, scores_used[:, :, -1].log(), torch.tensor(0.0, device=scores.device)) / (nomatching_criterion1.float().sum() + 1) 
        nomatching_loss_reverse = - torch.where(if_matching2, scores[:, -1, :-1].log(), torch.tensor(0.0, device=scores.device)) / (if_matching2.float().sum() + 1) 

        # 计算reverse

        data.update({
            'hw0_c': M, 'hw1_c': M,
            'hw0_f': 4, 'hw1_f': 4,
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "whole_loss": whole_loss.reshape(-1),
            "mkpts0_f_reverse": mkpts0_f_reverse,
            "mkpts1_f_reverse": mkpts1_f_reverse,
            "whole_loss_reverse": whole_loss_reverse.reshape(-1),
            "position_loss": position_loss.sum(),
            "average_error": average_error,
            "label_third_new": label,
            "nomatching_loss": nomatching_loss,
            "nomatching_loss_reverse": nomatching_loss_reverse
        })
        # 4. fine-level refinement
        # if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
        #     feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        # self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # a = data['mkpts1_f'].sum()
        # a.backward()
        # for name, params in self.backbone.named_parameters():
        #     print(name, params.grad)
        return data

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
    def loss(self, input_point, average, label, threthold = 2.0, label_dense=None):
        zero = torch.tensor(0.0, device=average.device).double()
        input_point_z = torch.ones([input_point.shape[0], 1], device=label.device)
        input_point = torch.cat([input_point, input_point_z], dim=1).double()
        output_point = torch.cat([average, input_point_z], dim=1).double()
        F = label.reshape(label.shape[0], 1, 3, 3).expand(-1, average.shape[0] // label.shape[0], -1, -1).reshape(-1, 3, 3)
        sum = torch.einsum('ik,ikl,il->i', output_point, F, input_point).abs()
        line1 = torch.einsum('ikl,il->ik', F, input_point)
        sampson_error = sum**2/((line1[:, :2] ** 2).sum(1) + 1e-7)
        distance =  sum/((line1[:, :2] ** 2 + 1e-7).sum(1).sqrt())
        sampson_error = torch.where(sampson_error < zero + threthold, sampson_error, zero + threthold)
        sampson_error = torch.where(distance < zero + 2, sampson_error * 2, sampson_error)
        sampson_error = torch.where(distance < zero + 1, sampson_error * 2, sampson_error)
        sampson_error = torch.where(distance < zero + 0.5, sampson_error * 2, sampson_error)
        sampson_error = torch.where(distance < zero + 0.25, sampson_error * 2, sampson_error)
        if label_dense!= None:
            sampson_error = torch.where(label_dense[:, :, 0].reshape(-1) > -9.9, sampson_error, zero)
        epipolar_loss = sampson_error.sum() / (average.shape[0] + 1e-8)
        return epipolar_loss, distance

    def Compute_result(self, scores, W, T, scale_x, scale_y, p_s, p_t, device):
        scores_back = self.pad(scores[:, :-1, :-1].reshape(scores.shape[0], scores.shape[1] - 1, W, W)).reshape(scores.shape[0], scores.shape[1] - 1, -1)
        scores_back = scores_back.reshape(scores.shape[0], W, W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1)
        # 提取T*T邻域， 并计算回归坐标结果
        max0 = scores[:, :-1, :-1].max(2)[1].reshape(scores.shape[0], W, W)[:, 2:6, 2:6].reshape(scores.shape[0], 16)
        x3 = (max0 % W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=device).reshape(1, 1, 1, T).repeat(max0.shape[0], max0.shape[1], T, 1).reshape(max0.shape[0], max0.shape[1], T**2)
        y3 = (max0 // W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=device).reshape(1, 1, T, 1).repeat(max0.shape[0], max0.shape[1], 1, T).reshape(max0.shape[0], max0.shape[1], T**2)
        index3 = y3 * (W + 4) + x3
        # print(index3.max(), index3.min())
        # import pdb
        # pdb.set_trace()
        scale_x_new = self.pad_1(scale_x.reshape(1, -1, W, W)).reshape(scale_x.shape[0], 1, -1).expand(-1, 16, -1)
        scale_y_new = self.pad_1(scale_y.reshape(1, -1, W, W)).reshape(scale_y.shape[0], 1, -1).expand(-1, 16, -1)
        scores_unfold_x = torch.gather((scores_back + 1e-7).sqrt() / scale_x_new, 2, index3)
        scores_unfold_y = torch.gather((scores_back + 1e-7).sqrt() / scale_y_new, 2, index3)
        positions = create_meshgrid(T, T, False, device=device).reshape(-1, 2) * 2 - (T - 1)
        weighted_point_x = torch.einsum('ijp,p->ij', scores_unfold_x, positions[:, 0])
        weighted_point_y = torch.einsum('ijp,p->ij', scores_unfold_y, positions[:, 1])
        point_weight_sum_x = scores_unfold_x.sum(2)
        point_weight_sum_y = scores_unfold_y.sum(2)
        # 计算真实坐标系下的坐标点
        mkpts1_f = torch.zeros([scale_x.shape[0], 16, 2], device=scores.device)
        mkpts1_f[:, :, 0] = weighted_point_x / point_weight_sum_x + (max0 % W + 0.5 - W/2) * 2.0
        mkpts1_f[:, :, 1] = weighted_point_y / point_weight_sum_y + (max0 // W + 0.5 - W/2) * 2.0
        mkpts1_f = mkpts1_f + p_t.reshape(-1, 1, 2).repeat(1, 16, 1)
        mkpts0_f = p_s.reshape(-1, 1, 2).repeat(1, 16, 1) + create_meshgrid(4, 4, False, device=device)\
            .reshape(1, -1, 2).repeat(p_s.shape[0], 1, 1) * 2 - 3.0
        # 计算whole_loss
        scores_unfold_sum = torch.gather(scores_back, 2, index3).sum(2)
        whole_loss = (scores[:, :-1, :].reshape(scores.shape[0], W, W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1).sum(2) - scores_unfold_sum)
            #  + torch.gather(scores[:, -1, :], 1, max0)
        whole_loss = torch.where(whole_loss >= 1e-2, whole_loss, torch.tensor(0.0, device=device)) / ((whole_loss >= 1e-2).float().sum() + 10) / 10
        return mkpts0_f, mkpts1_f, whole_loss

    def Compute_result_reverse(self, scores, W, T, scale_x, scale_y, p_s, p_t, device):
        scores_back = self.pad(scores[:, :-1, :-1].reshape(scores.shape[0], scores.shape[1] - 1, W, W)).reshape(scores.shape[0], scores.shape[1] - 1, -1)
        scores_back = scores_back.reshape(scores.shape[0], W, W, -1)[:, :, :, :].reshape(scores.shape[0], 64, -1)
        # 提取T*T邻域， 并计算回归坐标结果
        max0 = scores[:, :-1, :-1].max(2)[1].reshape(scores.shape[0], W, W)[:, :, :].reshape(scores.shape[0], 64)
        x3 = (max0 % W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=device).reshape(1, 1, 1, T).repeat(max0.shape[0], max0.shape[1], T, 1).reshape(max0.shape[0], max0.shape[1], T**2)
        y3 = (max0 // W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=device).reshape(1, 1, T, 1).repeat(max0.shape[0], max0.shape[1], 1, T).reshape(max0.shape[0], max0.shape[1], T**2)
        index3 = y3 * (W + 4) + x3
        # print(index3.max(), index3.min())
        # import pdb
        # pdb.set_trace()
        scale_x_new = self.pad_1(scale_x.reshape(1, -1, W, W)).reshape(scale_x.shape[0], 1, -1).expand(-1, 64, -1)
        scale_y_new = self.pad_1(scale_y.reshape(1, -1, W, W)).reshape(scale_y.shape[0], 1, -1).expand(-1, 64, -1)
        scores_unfold_x = torch.gather((scores_back + 1e-7).sqrt() / scale_x_new, 2, index3)
        scores_unfold_y = torch.gather((scores_back + 1e-7).sqrt() / scale_y_new, 2, index3)
        positions = create_meshgrid(T, T, False, device=device).reshape(-1, 2) * 2 - (T - 1)
        weighted_point_x = torch.einsum('ijp,p->ij', scores_unfold_x, positions[:, 0])
        weighted_point_y = torch.einsum('ijp,p->ij', scores_unfold_y, positions[:, 1])
        point_weight_sum_x = scores_unfold_x.sum(2)
        point_weight_sum_y = scores_unfold_y.sum(2)
        # 计算真实坐标系下的坐标点
        mkpts1_f = torch.zeros([scale_x.shape[0], 64, 2], device=scores.device)
        mkpts1_f[:, :, 0] = weighted_point_x / point_weight_sum_x + (max0 % W + 0.5 - W/2) * 2.0
        mkpts1_f[:, :, 1] = weighted_point_y / point_weight_sum_y + (max0 // W + 0.5 - W/2) * 2.0
        mkpts1_f = mkpts1_f + p_t.reshape(-1, 1, 2).repeat(1, 64, 1)
        mkpts0_f = p_s.reshape(-1, 1, 2).repeat(1, 64, 1) + create_meshgrid(8, 8, False, device=device)\
            .reshape(1, -1, 2).repeat(p_s.shape[0], 1, 1) * 2 - 7.0
        # 计算whole_loss
        scores_unfold_sum = torch.gather(scores_back, 2, index3).sum(2)
        whole_loss = (scores[:, :-1, :].reshape(scores.shape[0], W, W, -1)[:, :, :, :].reshape(scores.shape[0], 64, -1).sum(2) - scores_unfold_sum)
            #  + torch.gather(scores[:, -1, :], 1, max0)
        whole_loss = torch.where(whole_loss >= 1e-2, whole_loss, torch.tensor(0.0, device=device)) / ((whole_loss >= 1e-2).float().sum() + 10) / 10
        return mkpts0_f, mkpts1_f, whole_loss
