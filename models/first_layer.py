from asyncio import base_futures
import math
import random
from resnet import ResNet, BasicBlock
from datetime import datetime
from utils.utils import Compute_loss, project_function
import torchvision.models as models
from models.modules import *
import torchvision.transforms.functional as transforms
import torchvision
from .fine_preprocess import FinePreprocess
from .fine_matching import FineMatching
import time

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FPN_32_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/32 and 1/2.
    Each block has 2 layers.
    """
    def __init__(self):
        super().__init__()
        block_dims = [64, 64, 128, 256, 256]
        block_dims_before = [64, 64, 128, 256, 256]

        self.kenc4 = KeypointEncoder(256, [32, 64, 128, 256])
        self.kenc3 = KeypointEncoder(256, [32, 64, 128, 128])
        self.kenc2 = KeypointEncoder(128, [32, 64, 64, 64])
        self.kenc1 = KeypointEncoder(64, [32, 64, 64, 64])

        self.layer5_outconv = conv1x1(block_dims_before[4], block_dims[4])
        self.layer5_outconv2 = nn.Sequential(
            conv3x3(block_dims[4], block_dims[4]),
            nn.SyncBatchNorm(block_dims[4], momentum=0.001),
            nn.LeakyReLU(),
            conv3x3(block_dims[4], block_dims[4]),
        )

        self.layer4_outconv = conv1x1(block_dims_before[3], block_dims[4])
        self.layer4_outconv2 = nn.Sequential(
            conv3x3(block_dims[4], block_dims[4]),
            nn.SyncBatchNorm(block_dims[4], momentum=0.001),
            nn.LeakyReLU(),
            conv3x3(block_dims[4], block_dims[3]),
        )

        self.layer3_outconv = conv1x1(block_dims_before[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.SyncBatchNorm(block_dims[3], momentum=0.001),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )
        self.layer2_outconv = conv1x1(block_dims_before[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.SyncBatchNorm(block_dims[2], momentum=0.001),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims_before[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.SyncBatchNorm(block_dims[1], momentum=0.001),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_before):
        # outpout dims: (b, 9, h/8, w/8)
        b, f, h, w = x.shape
        cols = torch.arange(0, h * 16, device=x.device).reshape(h * 16, 1).repeat(1, w * 16).reshape(-1) / float(h * 16)
        rows = torch.arange(0, w * 16, device=x.device).reshape(1, w * 16).repeat(h * 16, 1).reshape(-1) / float(w * 16)
        kpts  = torch.zeros((h * w * 256), 2, device=x.device)
        kpts[:, 0] = cols
        kpts[:, 1] = rows
        kpts1 = self.kenc1(kpts)
        cols = torch.arange(0, h * 8, device=x.device).reshape(h * 8, 1).repeat(1, w * 8).reshape(-1) / float(h * 8)
        rows = torch.arange(0, w * 8, device=x.device).reshape(1, w * 8).repeat(h * 8, 1).reshape(-1) / float(w * 8)
        kpts  = torch.zeros((h * w * 64), 2, device=x.device)
        kpts[:, 0] = cols
        kpts[:, 1] = rows
        kpts2 = self.kenc2(kpts)
        cols = torch.arange(0, h * 4, device=x.device).reshape(h * 4, 1).repeat(1, w * 4).reshape(-1) / float(h * 4)
        rows = torch.arange(0, w * 4, device=x.device).reshape(1, w * 4).repeat(h * 4, 1).reshape(-1) / float(w * 4)
        kpts  = torch.zeros((h * w * 16), 2, device=x.device)
        kpts[:, 0] = cols
        kpts[:, 1] = rows
        kpts3 = self.kenc3(kpts)
        cols = torch.arange(0, h * 2, device=x.device).reshape(h * 2, 1).repeat(1, w * 2).reshape(-1) / float(h * 2)
        rows = torch.arange(0, w * 2, device=x.device).reshape(1, w * 2).repeat(h * 2, 1).reshape(-1) / float(w * 2)
        kpts  = torch.zeros((h * w * 4), 2, device=x.device)
        kpts[:, 0] = cols
        kpts[:, 1] = rows
        kpts4 = self.kenc4(kpts)
        x_out = self.layer5_outconv2(x) + self.layer5_outconv(x_before[-1])

        x5_out_2x = F.interpolate(x_out, scale_factor=2., mode='bilinear', align_corners=False)
        x4_out = self.layer4_outconv(x_before[-2])
        x_out = self.layer4_outconv2(x4_out+x5_out_2x + kpts4.reshape(1, -1, h * 2, w * 2))

        x4_out_2x = F.interpolate(x_out, scale_factor=2., mode='bilinear', align_corners=False)
        x3_out = self.layer3_outconv(x_before[-3])
        x_out1 = self.layer3_outconv2(x3_out+x4_out_2x + kpts3.reshape(1, -1, h * 4, w * 4))

        x3_out_2x = F.interpolate(x_out1, scale_factor=2., mode='bilinear', align_corners=False)
        x2_out = self.layer2_outconv(x_before[1])
        x_out = self.layer2_outconv2(x2_out+x3_out_2x + kpts2.reshape(1, -1, h * 8, w * 8))

        x2_out_2x = F.interpolate(x_out, scale_factor=2., mode='bilinear', align_corners=False)
        x1_out = self.layer1_outconv(x_before[0])
        x_out2 = self.layer1_outconv2(x1_out+x2_out_2x + kpts1.reshape(1, -1, h * 16, w * 16))
        return x_out1, x_out2
        # return x_out1



class SuperGlue_new(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'keypoint_encoder2': [8, 16, 32, 64],
        'GNN_layers': ['self', 'cross'] * 5,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self):
        super().__init__()
        self.config = {**self.default_config}
        self.descriptor_extract = ResNet(BasicBlock, [2, 2, 2, 2])
        pretrained_dict = models.resnet18(pretrained=True).state_dict()
        model_dict = self.descriptor_extract.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.descriptor_extract.load_state_dict(model_dict)
        # self.descriptor_extract.eval()
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.kenc_local = KeypointEncoder(
            self.config['descriptor_dim'] // 2, self.config['keypoint_encoder'])
        self.kenc_hypo = KeypointEncoder(
            self.config['descriptor_dim'] // 2, self.config['keypoint_encoder'])
        # for p in self.gnn.parameters():
        #    p.requires_grad = False
        # for p in self.final_proj.parameters():
        #    p.requires_grad = False
        self.softmax = nn.Softmax(dim=2)
        self.relu = torch.nn.ReLU()
        # for p in self.scalex_proj.parameters():
        #    p.requires_grad = False
        self.sigmoid = nn.Sigmoid()
        # bin_score = torch.nn.Parameter(torch.ones([self.default_config['descriptor_dim']]))
        self.local_segment = FPN_32_2()
        # self.ranges = self.ranges.reshape(1, 20, 20)
        assert self.config['weights'] in ['indoor', 'outdoor']
        
        self.compress = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)

        self.compress2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        # self.pos_encode = PositionEncodingSine(448)
        self.gather_homography = torch.nn.Conv2d(9, 9 * 9, (3, 3), stride=1, padding=1, bias=False)
        self.gather_homography.weight.data[:, :, :, :] = 0
        for i in range(3):
            for j in range(3):
                for k in range(9):
                    index = i * 27 + j * 9 + k
                    self.gather_homography.weight.data[index, k, i, j] = 1
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'], self.gather_homography)
        self.fine_preprocess = FinePreprocess()
        self.fine_matching = FineMatching()
        # for p in self.parameters():
        #    p.requires_grad = False
        # self.deformable_gnn = DeformableGNN(self.config['descriptor_dim'], self.config['Deformable_GNN_layers'], self.gather_homography)
        # for p in self.deformable_gnn.parameters():
        #    p.requires_grad = True
        for p in self.gather_homography.parameters():
           p.requires_grad = False
        # self.step1 = torch.compile(self.step1, mode='reduce-overhead', fullgraph=True)
        # self.gnn.forward = torch.compile(self.gnn.forward, mode='reduce-overhead', fullgraph=True)
        # self.fine_preprocess.forward = torch.compile(self.fine_preprocess.forward, mode='reduce-overhead', fullgraph=True)

    def step1(self, img): 
        b = img.shape[0] // 2
        self.one = torch.tensor(1.0, device=img.device)
        self.zeros = torch.tensor(0.0, device=img.device)
        desc_ = self.descriptor_extract.forward2(img)
        desc = desc_[-1]
        desc = self.compress(desc)
        desc_[-1] = desc
        h = desc.shape[2]
        w = desc.shape[3]
        kpts  = torch.zeros((desc.shape[2] * desc.shape[3]), 2, device=img.device)
        kpts[:, 0] = torch.arange(0, desc.shape[2], device=img.device).reshape(desc.shape[2], 1).repeat(1, desc.shape[3]).reshape(-1) / float(desc.shape[2])
        kpts[:, 1] = torch.arange(0, desc.shape[3], device=img.device).reshape(1, desc.shape[3]).repeat(desc.shape[2], 1).reshape(-1) / float(desc.shape[3])
        kpts1 = self.kenc(kpts)
        desc = (desc).reshape(2, b, self.config['descriptor_dim'], -1)
        desc0 = desc[0]
        desc1 = desc[1]
        # Multi-layer Transformer network.
        # Keypoint MLP encoder.
        return desc_, desc0, desc1, h, w, b, kpts1



    def forward(self, img, if_refine=False):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        with torch.no_grad():
            desc_, desc0, desc1, h, w, b, kpts1 = self.step1(img)
            # torch.cuda.synchronize()
            # start = time.time()
            desc0, desc1, results, _ = self.gnn(desc0, desc1, h, w, kpts1.reshape(-1, desc0.shape[1], desc0.shape[2]))

        # only for test
        project_results, grid = project_function(results[-1]["H_matrix"], w, h, 32)
        if_matching = results[-1]["confidence"].float() > 0.4
        # matching_coarse_tgt = torch.gather(project_results, 3, seg_result[:, :, :, None, None].expand(-1, -1, -1, -1, 3)).squeeze(3)
        matching_coarse_tgt = project_results.reshape(desc0.shape[0], h, 16, 
            w, 16, 9, 3).permute(0, 1, 3, 2, 4, 5, 6)[:, :, :, [3, 3, 11, 11], [3, 11, 3, 11], 4, :2].reshape(desc0.shape[0], 
            h * w, 4, 2)[if_matching]
        matching_coarse_ref = grid[:, :, :, [3, 3, 11, 11], [3, 11, 3, 11], :2].reshape(desc0.shape[0], h * w, 4, 2)[if_matching]
        results[-1]["mkpts0_f"] = matching_coarse_ref.reshape(-1, 2)
        results[-1]["mkpts1_f"] = matching_coarse_tgt.reshape(-1, 2)

        if if_refine:
            # torch.cuda.synchronize()
            # end = time.time()
            # print(end-start)
            grid_x, grid_y = torch.meshgrid(torch.arange(4, device=img.device) - 1.5, torch.arange(4, device=img.device) - 1.5)
            grid = torch.stack([grid_y, grid_x], -1).reshape(-1, 2).unsqueeze(0)
            desc_local = self.kenc_local(grid).reshape(1, -1, 1, 4, 1, 4).repeat(1, 1, h, 1, w, 1).reshape(1, -1, h * 4, w * 4)
            desc_8, desc_2 = self.local_segment(torch.cat([desc0, desc1], 0).reshape(-1, 256, h, w), [desc_[0], desc_[1], desc_[2], 
                desc_[3], desc_[4]])
            # desc_8 = self.local_segment(torch.cat([desc0, desc1], 0).reshape(-1, 256, h, w), [desc_[0], desc_[1], desc_[2], 
            #     desc_[3], desc_[4]])
            desc0_8, desc1_8, desc0_2, desc1_2 = desc_8[:b], desc_8[b:], desc_2[:b], desc_2[b:]
            # desc0_8, desc1_8 = desc_8[:left.shape[0]], desc_8[left.shape[0]:]
            # desc0_8 = (desc0_8 + desc_local).reshape(b, -1, h, 4, w, 4)
            grid_x, grid_y = torch.meshgrid(torch.arange(3, device=img.device) - 1, torch.arange(3, device=img.device) - 1)
            grid = torch.stack([grid_y, grid_x], -1).reshape(1, -1, 2).float()
            # desc0_hypo = self.gather_homography(self.compress2(desc0.reshape(b, -1, h, w)).reshape(-1, 1, h, w).expand(-1, 9, -1, -1)).reshape(\
            #     b, -1, 9, 9, h, w)[:, :, :, 0] + self.kenc_hypo(grid).reshape(1, -1, 9, 1, 1)
            #(b, 128, 9, h, w),(b. 128, h, 4, w, 4)->(b, 9, h, 4, w, 4)
            # scores_segment = torch.einsum('bfthw,bfhawc->bthawc', desc0_hypo, desc0_8).reshape(b, -1, h * 4, w * 4).sigmoid()
            # results[-1]['scores_segment'] = scores_segment
            # base_feature = desc_[0].reshape(2, left.shape[0], 64, h * 16, w * 16)
            # 选择要进行下一步的输入点坐标
            project_results, grid = project_function(results[-1]["H_matrix"], w, h, 32)
            project_results = project_results.reshape(-1, h, 4, 4, w, 4, 4, 9, 3)[:, :, :, 2, :, :, 2].reshape(-1, h * 4, w * 4, 9, 3)
            grid = grid.reshape(-1, h, w, 4, 4, 4, 4, 3)[:, :, :, :, 2, :, 2, :].permute(0, 1, 3, 2, 4, 5).reshape(-1, h * 4, w * 4, 3)
            # _, seg_result = scores_segment.max(1)
            seg_result = torch.zeros([b,h *4, w * 4], device=img.device).long() + 4
            confidence1 = self.gather_homography((results[-1]["confidence"]).float().reshape(-1, 1, h, w).expand(-1,
                9, -1, -1)).reshape(-1, 9, 9, h, w)[:, :, 0]
            b, _, h, w = confidence1.shape
            confidence1 = confidence1.permute(0, 2, 3, 1).unsqueeze(2).unsqueeze(4).expand(-1, -1, 4, -1, 4, -1).reshape(b, h * 4, w * 4, 9)
            if_matching = torch.gather(confidence1, 3, seg_result.unsqueeze(3)).squeeze(3) > 0.4
            # random_change = torch.ones_like(if_matching.float()).reshape(-1)
            # random_sequnce = torch.randperm(random_change.shape[0])
            # random_change[:int(random_change.shape[0])]
            matching_coarse_tgt = torch.gather(project_results, 3, seg_result[:, :, :, None, None].expand(-1, -1, -1, -1, 3)).squeeze(3)
            matching_coarse_tgt = matching_coarse_tgt[if_matching]
            matching_coarse_ref = grid[if_matching]
            bids = torch.arange(b, device=img.device)[:, None, None].expand(-1, h * 4, w * 4)
            bids = bids[if_matching]
            # matching_coarse_tgt = matching_coarse_tgt + torch.randn_like(matching_coarse_tgt)
            matching_coarse_tgt[:, 0] = torch.clamp(matching_coarse_tgt[:, 0], 0, w * 32 - 1)
            matching_coarse_tgt[:, 1] = torch.clamp(matching_coarse_tgt[:, 1], 0, h * 32 - 1)
            i_ids = (matching_coarse_ref[:, 0] / 8.0).long() + (matching_coarse_ref[:, 1] / 8.0).long() * w * 4
            j_ids = (matching_coarse_tgt[:, 0] / 2.0 + 3).long() + (matching_coarse_tgt[:, 1] / 2.0 + 3).long() * (w * 16 + 6) 
            data = {'b_ids': bids, 'i_ids': i_ids, "j_ids": j_ids, 'bs': b,
                'hw0_i': img.shape[1:3], 'hw1_i': img.shape[1:3], 'mkpts0_c': (matching_coarse_ref[..., :2].int() // 2).float().detach() * 2 + 1,
                'mkpts1_c': (matching_coarse_tgt[..., :2].int() // 2).float().detach() * 2 + 1}
            feat_ref, feat_tgt, confidence2 = self.fine_preprocess(desc0_2, desc1_2, desc0_8, desc1_8, data)
            self.fine_matching(feat_ref, feat_tgt, data)
            results[-1].update(data)
            # results[-1]["matches0"] = matching_coarse_ref[..., :2]
            # results[-1]["matches1"] = matching_coarse_tgt[..., :2]
            results[-1]["if_matching"] = if_matching
            results[-1]["confidence2"] = confidence2
            results[-1]["if_matching2"] = (confidence2 > 0.01)
        # results_second = self.deformable_gnn(desc_, desc0.reshape(desc0.shape[0], desc0.shape[1], h, w), desc1.reshape(desc0.shape[0], desc0.shape[1], h, w), bias, 
        #     results[-1]["confidence"] > 0.5, results[-1]['H_matrix'], project_results, h, w)
        # results_second[-1]["project_test"] = project_results
        # return results, results_second, self.gather_homography
        return results, self.gather_homography

    def loss_function(self, label1, label2, left, right, weight=[1.0, 1.0, 10.0, 10.0, 1.0], loss_type='distance', 
            eval_scores=None, refine_mode=True, if_choose = True, gather_homography=None, results = None,
            left_feature=None, right_feature=None):
        loss = Compute_loss(results, label1[:, :, 0:3], label2[:, :, 0:3],
                            left, right, refine_mode=refine_mode, loss_type=loss_type, eval_scores=eval_scores, if_choose=if_choose, 
                            gather_homography=gather_homography, left_feature=left_feature, right_feature=right_feature)
        return loss