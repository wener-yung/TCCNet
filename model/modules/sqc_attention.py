import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import *
from .fusion import *


class PCRA(nn.Module):
    def __init__(self, in_channel, m_channel=4, channel=32, t=3, depth=3, kernel_size=3, k = 8):
        super(PCRA, self).__init__()
        self.k = k
        self.in_channel = in_channel
        self.m_channel = m_channel
        self.channel = channel
        self.t = t
        self.pos_emb_l = PositionEmbeddingSine(num_pos_feats=in_channel / 2)
        self.pos_emb_h = PositionEmbeddingSine(num_pos_feats=in_channel / 2)
        self.query_conv = nn.Conv2d(self.in_channel, m_channel, 1, bias=False)
        self.key_conv = nn.Conv2d(self.in_channel, m_channel, 1, bias=False)

        self.conv_in = conv(channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, f, map, f_q, f_k, m_k):
        """
        Args:
            f: F_mem [b, c, h, w]
            map: prediction map from previous layer
            f_q: feature of current frame [b,c,h,w]
            f_k: feature of the reference frame  [b,c,h,w]
            m_k: gt of the reference frame
        Returns:

        """
        b, c, h, w = f_q.shape
        # map = F.interpolate(map, size=f.shape[-2:], mode='bilinear', align_corners=False)
        map = self.ret(map, f)
        m_k = self.ret(m_k, f)

        pos_q = self.pos_emb_h(f_q)
        pos_k = self.pos_emb_l(f_k)
        fq_emb = f_q + pos_q.contiguous()
        fk_emb = f_k + pos_k.contiguous()

        query = self.query_conv(fq_emb).view(b, self.m_channel, -1).permute(0, 2, 1)  # b, t*h*w, c
        key = self.key_conv(fk_emb).view(b, self.m_channel, -1)  # b, c, t*h*w
        map_g = torch.exp(m_k) / math.e
        value = map_g.view(b, 1, -1)

        sim = torch.bmm(query, key)  # b, thw, thw
        sim = F.softmax(sim, dim=-2)

        value = value.expand(-1, sim.shape[-2], -1)  # b, thw, thw
        context = sim.mul(value)
        context_k, _ = context.topk(k=self.k, dim=-1)
        context_m = torch.mean(context_k, dim=-1, keepdim=True)  # b, thw, 1

        corrtcted_map = context_m.view(b, h, w, 1).permute(0, 3, 1, 2)  # bt, 1, h, w

        corrtcted_map = corrtcted_map * map + map
        corrtcted_map_sig = torch.sigmoid(corrtcted_map)
        rmap = -1 * corrtcted_map_sig + 1  # bt, 1, h, w

        x = rmap.expand(-1, f.shape[1], -1, -1).mul(f)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + corrtcted_map
        return x, out, corrtcted_map


class SCRA(nn.Module):
    def __init__(self, in_channel_l, in_channel_h, m_channel=4, channel=32, t=3, depth=3, kernel_size=3, k = 8):
        super(SCRA, self).__init__()
        self.k = k
        self.in_channel_l = in_channel_l
        self.in_channel_h = in_channel_h
        self.m_channel = m_channel
        self.channel = channel
        self.t = t
        self.pos_emb_l = PositionEmbeddingSine(num_pos_feats=in_channel_l / 2)
        self.pos_emb_h = PositionEmbeddingSine(num_pos_feats=in_channel_h / 2)
        self.query_conv_l = nn.Conv3d(self.in_channel_l, m_channel, 1, bias=False)
        self.key_conv_l = nn.Conv3d(self.in_channel_l, m_channel, 1, bias=False)
        self.query_conv_h = nn.Conv3d(self.in_channel_h, m_channel, 1, bias=False)
        self.key_conv_h = nn.Conv3d(self.in_channel_h, m_channel, 1, bias=False)
        # self.value_conv = nn.Conv3d(self.in_channel, self.channel, 1, bias=False)

        self.conv_in = conv(channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, fl, fh, map):
        """
        fl: low level feature [bt, c, h, w] 2*3 32
        fh: high level feature []
        """
        map = F.interpolate(map, size=fl.shape[-2:], mode='bilinear', align_corners=False)
        pos_h = self.pos_emb_h(fh)
        pos_l = self.pos_emb_l(fl)
        crt_h = self.correct_map(fh, pos_h, map, self.query_conv_h, self.key_conv_h)
        crt_l = self.correct_map(fl, pos_l, map, self.query_conv_l, self.key_conv_l)
        corrtcted_map = (crt_l + crt_h) / 2

        corrtcted_map = corrtcted_map * map + map
        corrtcted_map_sig = torch.sigmoid(corrtcted_map)
        rmap = -1 * corrtcted_map_sig + 1  # bt, 1, h, w

        x = rmap.expand(-1, fl.shape[1], -1, -1).mul(fl)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + corrtcted_map
        return x, out, corrtcted_map

    def correct_map(self, f, pos, map, query_conv, key_conv):
        bt, c, h, w = f.shape
        b = bt // self.t
        f_emb = f + pos.contiguous()

        map_reshape = map.reshape(b, self.t, 1, h, w).permute(0, 2, 1, 3, 4) # b, 1, t, h, w
        map_reshape = torch.exp(torch.sigmoid(map_reshape)) / math.e

        f_reshape = f_emb.reshape(b, self.t, c, h, w).permute(0, 2, 1, 3, 4)  # b, c, t, h, w

        query = query_conv(f_reshape).view(b, self.m_channel, -1).permute(0, 2, 1)  # b, t*h*w, c
        key = key_conv(f_reshape).view(b, self.m_channel, -1)  # b, c, t*h*w
        value = map_reshape.view(b, 1, -1)  # b, 1, thw

        sim = torch.bmm(query, key)  # b, thw, thw
        sim = F.softmax(sim, dim=-2)

        value = value.expand(-1, sim.shape[-2], -1)  # b, thw, thw
        context = sim.mul(value)
        context_k, _ = context.topk(k=self.k, dim=-1)
        context_m = torch.mean(context_k, dim=-1, keepdim=True)  # b, thw, 1

        corrtcted_map = context_m.view(b, self.t, h, w, 1).view(b * self.t, h, w, 1).permute(0, 3, 1, 2)  # bt, 1, h, w
        return corrtcted_map.contiguous()


class PositionEmbeddingSine(nn.Module):
    """
    https://github.com/facebookresearch/detr
    """

    def __init__(self, num_pos_feats=16, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        # assert mask is not None
        b, c, h, w = x.shape
        # not_mask = torch.ones((b, h, w))  # ~mask
        not_mask = torch.ones((b, h, w)).cuda()
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t len = self.num_pos_feats
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)  # 10000 ^ 2 * i / dpos

        pos_x = x_embed[:, :, :, None] / dim_t  # [b, h, w, dim]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # [b, h, w, dim]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
