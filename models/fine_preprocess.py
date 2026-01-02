import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from utils.utils import project_function
from .linear_attention import FullAttention
from .modules import KeypointEncoder
import time

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)
        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        return x + message

class FinePreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_c_feat = False
        self.W1 = 3
        self.W2 = 7
        d_model_c = 128
        d_model_f = 64
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)
        # self.self_attention = LoFTREncoderLayer(64, 4)
        # self.cross_attention = LoFTREncoderLayer(64, 4)
        self.kenc = KeypointEncoder(64, [8, 16, 32, 64])
        self.kenc2 = KeypointEncoder(64, [8, 16, 32, 64])

        self.dim = d_model_f // 4
        self.nhead = 4

        # multi-head attention
        # self.qkv_proj_self = nn.Linear(d_model_f, d_model_f, bias=False)
        # self.self_attention = FullAttention()
        # self.merge_self = nn.Linear(d_model_f, d_model_f, bias=False)

        # # feed-forward network
        # self.mlp_self = nn.Sequential(
        #     nn.Linear(d_model_f*2, d_model_f, bias=False),
        #     nn.ReLU(True),
        #     nn.Linear(d_model_f, d_model_f, bias=False),
        # )
        # # norm and dropout
        # self.norm1_self = nn.SyncBatchNorm(d_model_f, momentum=0.001)
        # self.norm2_self = nn.SyncBatchNorm(d_model_f, momentum=0.001)

        # multi-head attention
        self.q_proj_cross = nn.Linear(d_model_f, d_model_f, bias=False)
        self.kv_proj_cross = nn.Linear(d_model_f, d_model_f, bias=False)
        self.cross_attention = FullAttention()
        self.merge_cross = nn.Linear(d_model_f, d_model_f, bias=False)

        # feed-forward network
        self.mlp_cross = nn.Sequential(
            nn.Linear(d_model_f * 2, d_model_f, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model_f, d_model_f + 1, bias=False),
        )
        # norm and dropout
        # self.norm1_cross = nn.SyncBatchNorm(d_model_f, momentum=0.001)
        # self.norm2_cross = nn.SyncBatchNorm(d_model_f, momentum=0.001)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W1 = self.W1
        W2 = self.W2
        stride = 4
        b, f, h, w = feat_f0.shape
        data.update({'W': W2})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W2**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W2**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1, feat1[:, :, -1].reshape(-1).sigmoid()
        kpts = torch.zeros((9), 2, device=feat_f0.device)
        kpts[:, 0] = torch.arange(-1, 2, device=feat_f0.device).reshape(3, 1).repeat(1, 3).reshape(-1)
        kpts[:, 1] = torch.arange(-1, 2, device=feat_f0.device).reshape(1, 3).repeat(3, 1).reshape(-1)
        # 1. unfold(crop) all local windows (b * l. ww, f)
        # print(F.unfold(feat_f0, kernel_size=(W1, W1), stride=stride, padding=W1//2).shape, feat_f0.shape, h, w)
        feat0 = feat_f0.permute(0, 2, 3, 1).reshape(b * h * w, f)
        feat_f0_query0 = feat0.reshape(b, h // 4, 4, w // 4, 4, f)[:, :, 2, :, 2].reshape(b, h * w // 16, f)
        # feat_f0_qkv = self.qkv_proj_self(feat0)
        # feat_f0_query = feat_f0_qkv[:, :f]
        # feat_f0_unfold = F.pad(feat_f0_qkv[..., :f].reshape(b, h, w, f), [0, 0, 0, 1, 0, 1])
        # feat_f0_unfold = F.unfold(feat_f0_unfold.permute(0, 3, 2, 1), kernel_size=(W1 + 2, W1 + 2), stride=stride).reshape(b,
        #     f, 25, h * w // 16)[:, :, [0, 4, 12, 20, 24]].permute(0, 3, 2, 1).flatten(0, 1)
        # feat_f0_key = feat_f0_unfold[..., :f].reshape(b * h * w // 16, 5, f)
        # feat_f0_value = feat_f0_key
        # feat_f0_query = feat_f0_query.reshape(b, h // 4, 4, w // 4, 4, f)[:, :, 2, :, 2].reshape(b * h * w // 16, 1, f)
        # # (b,f,w,w)
        # feat_f0_query = self.self_attention(feat_f0_query.view(-1, 1, self.nhead, self.dim), feat_f0_key.view(-1, 5, self.nhead, 
        #     self.dim), feat_f0_value.view(-1, 5, self.nhead, self.dim))
        # feat_f0_query = self.merge_self(feat_f0_query.view(b, -1, self.nhead*self.dim))  # [N, L, C]
        # feat_f0_query = self.norm1_self(feat_f0_query.permute(0, 2, 1)).permute(0, 2, 1)
        # # feed-forward network
        # # print(feat_f0_query.shape, feat0.shape)
        # feat_f0_query = self.mlp_self(torch.cat([feat_f0_query0, feat_f0_query], dim=2))
        # feat_f0_query = (self.norm2_self(feat_f0_query.permute(0, 2, 1)).permute(0, 2, 1) + feat_f0_query0.reshape(feat_f0_query.shape)).reshape(b, h * w // 16, f)
        
        # feat_f0_unfold = F.unfold(feat_f0_unfold, kernel_size=(W2, W2), stride=1, padding=W2//2)
        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W2**2)
        # torch.cuda.synchronize()
        # start = time.time()
        feat1 = feat_f1.permute(0, 2, 3, 1).reshape(b * h * w, f)
        # feat_f1_qkv = self.qkv_proj_self(feat1)
        # feat_f1_query = feat_f1_qkv[:, :f]
        # feat_f1_unfold = F.unfold(feat_f1_qkv[..., :f].reshape(b, h, w, f), kernel_size=(W1 + 2, W1 + 2), stride=1, padding=2).reshape(b,
        #     f, 25, h * w)[:, :, [0, 4, 12, 20, 24]].permute(0, 3, 2, 1).flatten(0, 1)
        # feat_f1_key = feat_f1_unfold[..., :f].reshape(b * h * w, 5, f)
        # feat_f1_value = feat_f1_key
        # feat_f1_query = feat_f1_query.reshape(b * h * w, 1, f)
        # # (b,f,w,w)
        # # print(self.self_attention(feat_f0_unfold[:, 12, None], feat_f0_unfold).shape)
        # feat_f1_query = self.self_attention(feat_f1_query.view(-1, 1, self.nhead, self.dim), feat_f1_key.view(-1, 5, self.nhead, 
        #     self.dim), feat_f1_value.view(-1, 5, self.nhead, self.dim))
        # feat_f1_query = self.merge_self(feat_f1_query.view(b, -1, self.nhead*self.dim))  # [N, L, C]
        # feat_f1_query = self.norm1_self(feat_f1_query.permute(0, 2, 1)).permute(0, 2, 1)s
        # # feed-forward network
        # feat_f1_query = self.mlp_self(torch.cat([feat1.reshape(feat_f1_query.shape), feat_f1_query], dim=2))
        # feat_f1_query = (self.norm2_self(feat_f1_query.permute(0, 2, 1)).permute(0, 2, 1) + feat_f1.reshape(feat_f1_query.shape)).reshape(b, h, w, f)
        cols = torch.arange(-3, 4, device=feat_f0.device).reshape(7, 1).repeat(1, 7).reshape(-1)
        rows = torch.arange(-3, 4, device=feat_f0.device).reshape(1, 7).repeat(7, 1).reshape(-1)
        kpts_new = torch.zeros((49), 2, device=feat_f0.device)
        kpts_new[:, 0] = rows
        kpts_new[:, 1] = cols
        torch.cuda.synchronize()
        start = time.time()
        feat_f1_kv = self.kv_proj_cross(feat_f1.reshape(-1, f)).reshape(b, h, w, f)
        feat_f1_kv = F.pad(feat_f1_kv, [0, 0, W2 // 2, W2 // 2, W2 // 2, W2 // 2]).reshape(-1, f)
        j_ids_new = (data['j_ids'] + data['b_ids'] * (h + 6) * (w + 6)).reshape(-1, 1) + kpts_new[None, :, 0] + kpts_new[None, :, 1] * (w + 6)

        # print(end-start)sss
        # 2. select only the predicted matches
        feat_f0_query = feat_f0_query0[data['b_ids'].long(), data['i_ids'].long()][:, None]  # [n, ww, cf]
        feat_f1_kv = feat_f1_kv[j_ids_new.long().reshape(-1)].reshape(-1, W2 * W2, f) + self.kenc2(kpts_new).permute(0, 2, 1)
        feat_f1_key = feat_f1_kv[..., :f].reshape(-1, W2 * W2, f)
        feat_f1_value = feat_f1_key
        feat_f0_query2 = self.q_proj_cross(feat_f0_query.reshape(-1, f)).reshape(-1, 1, f)
        # print(feat_f0_unfold.shape, feat_f1_unfold.shape)

        feat_f0_query2 = self.cross_attention(feat_f0_query2.view(-1, 1, self.nhead, self.dim), feat_f1_key.view(-1, W2 * W2, self.nhead, 
            self.dim), feat_f1_value.view(-1, W2 * W2, self.nhead, self.dim))
        # torch.cuda.synchronize()
        # end = time.time()
        # print(end-start)
        feat_f0_query2 = self.merge_cross(feat_f0_query2.view(-1, 1, self.nhead*self.dim))
        # feed-forward network
        feat_f0_query2 = self.mlp_cross(torch.cat([feat_f0_query, feat_f0_query2], dim=2))
        torch.cuda.synchronize()
        end = time.time()
        # print(end-start)
        confidence_refine = feat_f0_query2[:, :, -1].reshape(-1).sigmoid()
        feat_f0_query2 = feat_f0_query2[:, :, :f] + feat_f0_query.reshape(feat_f0_query2[:, :, :f].shape)
        # option: use coarse-level loftr feature as context: concat and linear
        feat_f1_query = F.pad(feat1.reshape(b, h, w, f), [0, 0, W2 // 2, W2 // 2, W2 // 2, W2 // 2]).reshape(-1, f)
        return feat_f0_query2, feat_f1_query[j_ids_new.long().reshape(-1)].reshape(-1, W2 * W2, f) + self.kenc2(kpts_new).permute(0, 2, 1), confidence_refine
        # return feat_f0_query2, feat_f1_kv
