import math

from matplotlib.pyplot import sca
from skimage import _INPLACE_MSG
from resnet import ResNet2, BasicBlock
from datetime import datetime
from utils.utils import Compute_loss, Iterative_expand_matrix
import torchvision.models as models
from models.modules import *
import torchvision

class Refine_module(nn.Module):
    default_config = {
        'descriptor_dim': 264,
        'weights': 'outdoor',
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'point_num': 144,
        'scores_ratio': [0.4, 0.3, 0.3],
        'scores_num': [20, 7, 2]
    }
    def __init__(self):
        super().__init__()
        self.row_num = 12
        self.config = {**self.default_config}
        self.descriptor_extract = ResNet2(BasicBlock, [3, 4, 6, 3])
        pretrained_dict = models.resnet34(pretrained=True).state_dict()
        model_dict = self.descriptor_extract.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.descriptor_extract.load_state_dict(model_dict)
        for p in self.descriptor_extract.parameters():
           p.requires_grad = True
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
        self.relu = torch.nn.ReLU(inplace=False)
        cols = torch.arange(0, self.row_num).reshape(self.row_num, 1).repeat(1, self.row_num).reshape(self.config['point_num'])
        rows = torch.arange(0, self.row_num).reshape(1, self.row_num).repeat(self.row_num, 1).reshape(self.config['point_num'])
        self.positions = torch.zeros((self.config['point_num'], 2))
        self.positions[:, 0] = cols
        self.positions[:, 1] = rows
        # self.ranges = self.ranges.reshape(1, 20, 20)
        assert self.config['weights'] in ['indoor', 'outdoor']
        self.upsampling = nn.Upsample(size=[12, 12], mode='bilinear', align_corners=True)
        self.avgpool = nn.AvgPool2d(2, stride=1, padding=1)
        # self.evaluate1 = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        # self.evaluate2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True)
        # self.evaluate3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, bias=True)
        # 3是平均以后评价batch的， 2是patch内部进行均衡的
        # self.evaluate4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        # self.evaluate5 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1, stride=1, bias=True)
        self.compress_1 = MLP([448, 256, 128, 64, 32, 16, 8])
        self.compress_2 = MLP([448, 448, 448, self.config['descriptor_dim']])
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def forward(self, left, right, desc_l, scales):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # left = left.to(self.device)
        # right = right.to(self.device)
        self.one = torch.tensor(1.0, device=left.device)
        self.zeros = torch.tensor(0.0, device=left.device)
        self.positions = self.positions.to(left.device).contiguous()
        # left = left.permute(0, 3, 1, 2).float().contiguous()
        # right = right.permute(0, 3, 1, 2).float().contiguous()
        left = self.normalize(left.permute(0, 3, 1, 2).float().contiguous())
        right = self.normalize(right.permute(0, 3, 1, 2).float().contiguous())
        pic0 = torch.cat([left, right], dim=0).reshape(-1, left.shape[1], left.shape[2], left.shape[3])
        desc0_ = self.descriptor_extract.forward2(pic0)
        desc = []
        for i, feat in enumerate(desc0_):
            stride = int(8.0 / torch.pow(torch.tensor(2.0, device=left.device), i + 1))
            # feat = feat[:, :, stride*2:stride*10, stride*2:stride*10]
            if i <= 1:
                feat = self.avgpool(feat)
            index = ((self.positions.reshape(self.row_num, self.row_num, 2) + 0.5) * stride).long()
            index = (index[:, :, 0] * feat.shape[3] + index[:, :, 1]).reshape(1, 1, -1).\
                repeat(feat.shape[0], feat.shape[1], 1)
            desc.append(torch.gather(feat.reshape(feat.shape[0], feat.shape[1], -1), 2, index))
        desc = torch.cat(desc, dim=1).reshape(2, left.shape[0], 256, -1)
        # desc = desc0_[2].reshape(2, left.shape[0], 128, -1)
        title = self.compress_1(desc_l.unsqueeze(2)).repeat(2, 1, self.config['point_num']).reshape(2, left.shape[0], 8, -1)
        rubbish = self.compress_2(desc_l.unsqueeze(2)).repeat(2, 1, 1).reshape(2, left.shape[0], self.config['descriptor_dim'], 1)
        desc = torch.cat([title, desc], dim=2)
        desc = torch.cat([desc ,rubbish], dim=3)
        desc0 = desc[0]
        desc1 = desc[1]
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scale_x = self.scalex_proj(mdesc1[:, :, :-1].reshape(mdesc1.shape[0], -1, self.row_num, self.row_num))\
            .reshape(right.shape[0], -1, self.config['point_num'])
        scale_y = self.scaley_proj(mdesc1[:, :, :-1].reshape(mdesc1.shape[0], -1, self.row_num, self.row_num))\
            .reshape(right.shape[0], -1, self.config['point_num'])
        scale_x = torch.exp(self.sigmoid(scale_x) * math.log(256.0) - math.log(256.0) / 2)
        scale_y = torch.exp(self.sigmoid(scale_y) * math.log(256.0) - math.log(256.0) / 2)
        scale = scale_x * scale_y
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        # scores = self.softmax(scores)
        # scores = F.normalize(scores, p=1, dim=2)
        scores = scores / self.config['descriptor_dim']**.5
        # print(scores)
        # Run the optimal transport.
        scores = log_optimal_transport2(
            0.1 * scores, self.one, scale,
            iters=self.config['sinkhorn_iterations'])
        # Get the matches with score above "match_threshold".
        # scale = scores[:, :-1, :-1].exp().sum(1)
        # with torch.no_grad():

        # scales = scales.mean()
        # if scales > 1:
        #     scales = scales * 2
        # else:
        #     scales = scales / 2

        # scores[:, :, -1] += (scales.reshape(-1, 1).repeat(1, 145)).log()
        # scores[:, -1, :] += (scales.reshape(-1, 1).repeat(1, 145)).log()
        # scores[:, -1, :] -= scores[:, :-1, :].max(1)[0]

        # scores[:, :, -1] += torch.log(self.one * 2)
        # scores[:, -1, :] += torch.log(self.one * 2)
        # scores[:, :, -1] += torch.log(self.one * 2)
        # scores[:, -1, :] += torch.log(self.one * 2)
        # scores[:, :, -1] += torch.log(self.one * 16)
        # scores[:, -1, :] += torch.log(self.one * 4)


        max0, max1 = scores[:, :, :self.config['point_num']].max(2), \
                     scores[:, :, :self.config['point_num']].max(1)
        # if_matching = torch.tensor(max0 != self.config['point_num'], device=scores.device)
        mdesc = mdesc0[:, :, :-1]
        # eval_scores = F.relu(self.evaluate2(F.relu(self.evaluate1(mdesc.reshape(mdesc.shape[0], -1, self.row_num, self.row_num)))))
        # eval_scores = F.relu(self.evaluate3(eval_scores))
        # eval_scores_positive = self.softmax(self.evaluate4(eval_scores)).reshape(mdesc.shape[0], -1, self.config['point_num']).permute(0, 2, 1)
        # eval_scores_negative = self.sigmoid(self.evaluate5(eval_scores)).reshape(mdesc.shape[0], -1, self.config['point_num']).permute(0, 2, 1) * 100
        # eval_scores_positive = torch.where(if_matching, eval_scores_positive,  self.zeros + 100.1)
        # eval_scores_negative = torch.where(if_matching, eval_scores_negative,  self.zeros - 0.1)
        # eval_scores = torch.cat([eval_scores_positive[:, :, 0].unsqueeze(2), eval_scores_positive[:, :, 1].unsqueeze(2), 
        #     eval_scores_negative[:, :, 0].unsqueeze(2), eval_scores_positive[:, :, 1].unsqueeze(2), eval_scores_negative[:, :, 1].unsqueeze(2)], dim=2)
        # whole_scores = (torch.tensor(self.config['scores_ratio'], device = scores.device) * left.shape[0]).int()
        # whole_scores[2] = left.shape[0] - whole_scores[0] - whole_scores[1]
        # whole_scores_origin = log_optimal_transport_eval(eval_scores[:left.shape[0], :, 0:3].mean(dim=1).unsqueeze(0), whole_scores.unsqueeze(0), 
        #     iters=self.config['sinkhorn_iterations']).squeeze(0).exp()
        # whole_scores_reverse = log_optimal_transport_eval(eval_scores[left.shape[0]:, :, 0:3].mean(dim=1).unsqueeze(0), whole_scores.unsqueeze(0), 
        #     iters=self.config['sinkhorn_iterations']).squeeze(0).exp()
        # whole_scores = torch.cat([whole_scores_origin, whole_scores_reverse], dim=0)
        # whole_scores = whole_scores.mm(torch.tensor(self.config['scores_num'], device=scores.device).float().reshape(3, 1)).repeat(1, 2)
        # 这里的0位置指的是有匹配的点的分数，所以直接把1舍弃掉也就可以了
        # whole_scores[:, 1] = self.config['point_num'] - whole_scores[:, 0]
        # result_scores = log_optimal_transport_eval(eval_scores[:, :, 3:5], whole_scores, iters=self.config['sinkhorn_iterations'])[:, :, 0].exp()
        result = [max0, max1]
        mdesc = torch.cat([mdesc0, mdesc1], dim=0)
        return result, scores, scale_x.squeeze(1), scale_y.squeeze(1), mdesc, desc0_, None

    def loss_function(self, scores, scalex, scaley, label1, label2, left, right,
                      weight=[1.0, 1.0, 10.0, 10.0, 1.0], eval_scores=None, loss_type='distance', row_num=12, if_choose=True):
        cols = torch.arange(0, row_num).reshape(row_num, 1).repeat(1, row_num).reshape(-1)
        rows = torch.arange(0, row_num).reshape(1, row_num).repeat(row_num, 1).reshape(-1)
        positions = torch.zeros((row_num * row_num , 2), device=scores.device)
        positions[:, 0] = cols
        positions[:, 1] = rows
        ranges = torch.zeros([row_num, row_num], device=scores.device)
        if loss_type == "homography":
            loss_type = "distance"
        for i in range(row_num):
            ranges[i] = F.pad(torch.arange(i + 1), [0, row_num - 1 - i], "constant", 1e7)
        max0, max1 = scores.max(2).indices, scores.max(1).indices
        loss = Compute_loss(scores, scalex, scaley, max0, max1, label1[:, :, 0:3], label2[:, :, 0:3],
            ranges, positions, left, right, weight, width=row_num, height=row_num, patch_scale=8, iter=8,
                eval_scores=eval_scores, loss_type=loss_type, reverse_set=True, if_refine=True, if_choose=if_choose)
        return loss
