import math
from resnet import ResNet3, BasicBlock
from datetime import datetime
from utils.utils import Compute_loss, Iterative_expand_matrix
import torchvision.models as models
from models.modules import *

class Third_module(nn.Module):
    default_config = {
        'descriptor_dim': 96,
        'weights': 'outdoor',
        'keypoint_encoder': [16, 32],
        'GNN_layers': ['self', 'cross'] * 3,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'point_num': 144
    }
    def __init__(self):
        super().__init__()
        self.config = {**self.default_config}
        # self.descriptor_extract = feature_extractor(layer_num=1, first_dim=32)
        self.descriptor_extract = ResNet3(BasicBlock, [3, 4, 6, 3])
        # pretrained_dict = models.resnet34(pretrained=True).state_dict()
        # model_dict = self.descriptor_extract.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.descriptor_extract.load_state_dict(model_dict)
        for p in self.descriptor_extract.parameters():
           p.requires_grad = True
        self.kenc = KeypointEncoder(
            64, self.config['keypoint_encoder'])
        self.sigmoid = nn.Sigmoid()
        self.scalex_proj = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=1,
                                     kernel_size=3, padding=1, stride=1, bias=True)
        self.scaley_proj = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=1,
                                     kernel_size=3, padding=1, stride=1, bias=True)
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.relu = torch.nn.ReLU()
        bin_score = torch.nn.Parameter(torch.tensor(0.0))
        self.register_parameter('bin_score', bin_score)
        cols = torch.arange(0, 12).reshape(12, 1).repeat(1, 12).reshape(self.config['point_num'])
        rows = torch.arange(0, 12).reshape(1, 12).repeat(12, 1).reshape(self.config['point_num'])
        self.positions = torch.zeros((self.config['point_num'], 2))
        self.positions[:, 0] = cols
        self.positions[:, 1] = rows
        # self.ranges = self.ranges.reshape(1, 20, 20)
        assert self.config['weights'] in ['indoor', 'outdoor']
        self.upsampling = nn.Upsample(size=[12, 12], mode='bilinear', align_corners=True)
        self.compress_0 = nn.Conv1d(in_channels=160, out_channels=32, kernel_size=1, padding=0, stride=1, bias=True)
        self.evaluate1 = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=40, kernel_size=3, padding=1, stride=1, bias=True)
        self.evaluate2 = nn.Conv2d(in_channels=40, out_channels=20, kernel_size=3, padding=1, stride=1, bias=True)
        self.evaluate3 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, padding=1, stride=1, bias=True)
        self.evaluate4 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        # self.compress_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, left, right, desc_l):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # left = left.to(self.device)
        # right = right.to(self.device)
        self.one = torch.tensor(1.0, device=left.device)
        self.zeros = torch.tensor(0.0, device=left.device)
        self.positions = self.positions.to(left.device).contiguous()
        left = left.contiguous()
        right = right.contiguous()
        pic0 = torch.cat([left, right], dim=0).reshape(-1, left.shape[1], left.shape[2], left.shape[3])
        desc0_ = self.descriptor_extract(pic0)
        # desc0_ = self.compress_1(desc0_)
        # desc0_ = self.test_conv(desc0_)
        desc = desc0_.reshape(pic0.shape[0], 64, -1)
        # Keypoint normalization.
        kpts = self.positions.to(left.device)
        # Keypoint MLP encoder.
        desc = (desc + self.kenc(kpts)).reshape(2, left.shape[0], 64, -1)
        desc0 = desc[0]
        desc1 = desc[1]
        title = self.compress_0(desc_l.unsqueeze(2)).repeat(1, 1, self.config['point_num'])
        desc0 = torch.cat([title, desc0], dim=1)
        desc1 = torch.cat([title, desc1], dim=1)
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scale_x = self.scalex_proj(mdesc1[:, :, :].reshape(mdesc1.shape[0], -1, 12, 12))\
            .reshape(right.shape[0], -1, self.config['point_num'])
        scale_y = self.scaley_proj(mdesc1[:, :, :].reshape(mdesc1.shape[0], -1, 12, 12))\
            .reshape(right.shape[0], -1, self.config['point_num'])
        scale_x = torch.exp(self.sigmoid(scale_x) * math.log(4.0) - math.log(4.0) / 2)
        scale_y = torch.exp(self.sigmoid(scale_y) * math.log(4.0) - math.log(4.0) / 2)
        scale = scale_x * scale_y
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        # scores = self.softmax(scores)
        # scores = F.normalize(scores, p=1, dim=2)
        scores = scores / self.config['descriptor_dim']**.5
        # print(scores)
        # Run the optimal transport.
        scores = log_optimal_transport(
            0.1 * scores, self.bin_score, self.one, scale,
            iters=self.config['sinkhorn_iterations'])
        # Get the matches with score above "match_threshold".
        # with torch.no_grad():
        eval_scores = F.relu(self.evaluate2(F.relu(self.evaluate1(mdesc0.reshape(mdesc1.shape[0], -1, 12, 12)))))
        eval_scores = F.softmax(self.evaluate4(F.relu(self.evaluate3(eval_scores))), dim=1).reshape(mdesc1.shape[0], -1, 144).permute(0, 2, 1)
        max0, max1 = scores[:, :self.config['point_num'], :self.config['point_num']].max(2), \
                     scores[:, :self.config['point_num'], :self.config['point_num']].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mscores0 = max0.values.exp()
        mscores1 = mscores0.gather(1, indices1)
        valid0 = (mscores0 > self.config['match_threshold'])
        valid1 = valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        result = {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
        return result, scores, scale_x, scale_y, eval_scores

    def loss_function(self, scores, scalex, scaley, label1, label2, left, right,
                      weight=[1.0, 1.0, 10.0, 10.0, 1.0], eval_scores=None, loss_type='distance'):
        cols = torch.arange(0, 12).reshape(12, 1).repeat(1, 12).reshape(144)
        rows = torch.arange(0, 12).reshape(1, 12).repeat(12, 1).reshape(144)
        positions = torch.zeros((144, 2), device=scores.device)
        positions[:, 0] = cols
        positions[:, 1] = rows
        ranges = torch.zeros([12, 12], device=scores.device)
        if loss_type == "homography":
            loss_type = "distance"
        for i in range(12):
            ranges[i] = F.pad(torch.arange(i + 1), [0, 11 - i], "constant", 1e7)
        max0, max1 = scores.max(2).indices, scores.max(1).indices
        loss = Compute_loss(scores, scalex, scaley, max0, max1, label1[:, :, 0:2], label2[:, :, 0:2],
            ranges, positions, left, right, weight, width=12, height=12, patch_scale=8, iter=25,
                eval_scores=eval_scores, loss_type=loss_type, reverse_set=True, if_refine=True, if_third=True)
        return loss
