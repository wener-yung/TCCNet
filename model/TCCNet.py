from __future__ import division
import math
import copy
import torch.optim as optim
from model.Res2Net_v1b import res2net50_v1b_26w_4s

from model.modules.fusion import Fusion
from model.modules.context_module import *
from model.modules.decoder_module import *
from model.modules.sqc_attention import SCRA, PCRA
from model.context_free.makecut import makebox, makebox_supervise
from model.context_free.get_location import *
from model.context_free.cf_loss import syn_loss
from model.losses import bce_iou_loss

from utils.tools import visualize
import torch.nn as nn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class TCCNet(nn.Module):
    def __init__(self, config):
        super(TCCNet, self).__init__()
        # encoder
        self.seg_encoder = Encoder()
        self.prop_encoder = None
        self._get_prop_encoder()

        # seg_branch
        self.seg_decoder = Decoder_sqc(32, k=config.k_corrected)

        # prop_branch
        self.conv_m = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.KV_M_r4 = KeyValue(32, keydim=4, valdim=16)
        self.KV_M_r3 = KeyValue(32, keydim=4, valdim=16)
        self.KV_M_r2 = KeyValue(32, keydim=4, valdim=16)
        self.Memory = Memory()
        self.prop_decoder = Decoder_prop_map(32, k=config.k_corrected)

        # optimizer and loss
        self.optim = None
        self.total_step = 1
        self.cut_loss = syn_loss()
        self.ceiou_loss = bce_iou_loss
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.config = config
        self.T = self.config.memory_size
        self.base_size = self.config.size

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def get_optimizer(self, seg_lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value]}]
        self.optim = optim.Adam(params, lr=seg_lr)
        return self.optim

    def update_target(self):
        self._get_prop_encoder()

    def _get_prop_encoder(self):
        if self.prop_encoder is None:
            self.prop_encoder = copy.deepcopy(self.seg_encoder)
        else:
            self._momentum_update_net(self.seg_encoder, self.prop_encoder)
        set_requires_grad(self.prop_encoder, False)

    def _momentum_update_net(self, net_q, net_k, m=0.99):
        """
        Momentum update of the key net
        """
        alpha = min(1 - 1 / (self.total_step + 1), m)
        if net_q is not None and net_k is not None:
            for param_q, param_k in zip(net_q.parameters(), net_k.parameters()):
                param_k.data = param_k.data * alpha + param_q.data * (1. - alpha)
        else:
            pass
        self.total_step += 1

    def context_free_pretrain(self, frame, label, border, device, epoch=0, idxx=0):
        dic1, dic2 = makebox_supervise(frame, border, label, epoch=epoch, idxx=idxx, config=self.config)
        x1, x2 = dic1['view'], dic2['view']
        y1, y2 = dic1['label'], dic2['label']
        x1, x2 = x1.to(device), x2.to(device)
        y1, y2 = y1.to(device), y2.to(device)

        seg_pred1, a1, _, _ = self.seg_encoder(x1)
        seg_pred2, a2, _, _ = self.seg_encoder(x2)

        seg_pred1 = self.res(seg_pred1, self.base_size)
        seg_pred2 = self.res(seg_pred2, self.base_size)
        loss1 = self.ceiou_loss(seg_pred1, y1)
        loss2 = self.ceiou_loss(seg_pred2, y2)

        if epoch < -1 and epoch % 20 == 0 and idxx % 8 == 0:
            dic1 = {
                'frame': frame[0], 'gt': label[0, 0],
                "x1": x1[0], "pred1": torch.sigmoid(seg_pred1[0, 0]), "y1": y1[0, 0],
                "x2": x2[0], "pred2": torch.sigmoid(seg_pred2[0, 0]), "y2": y2[0, 0],
            }
            visualize(dic1, self.config.visualize_path, epoch, idxx)

        return loss1 + loss2

    def context_free(self, frame, loaction, border, label, device, epoch=0, idxx=0):
        dic1, dic2 = makebox(frame, loaction, border, label, epoch=epoch, idxx=idxx, config=self.config)
        x1, x2 = dic1['view'], dic2['view']
        x1, x2 = x1.to(device), x2.to(device)

        seg_pred1, a1, _, _ = self.seg_encoder(x1)
        seg_pred2, a2, _, _ = self.seg_encoder(x2)

        with torch.no_grad():
            prop_pred1, prop_a1, _, _ = self.prop_encoder(x1)
            prop_pred2, prop_a2, _, _ = self.prop_encoder(x2)
            prop_pred1.detach_()
            prop_pred2.detach_()

        loss1 = self.cut_loss(self.res(seg_pred1, self.base_size),
                              self.res(prop_pred2, self.base_size),
                              dic1, dic2, device, epoch=epoch, idxx=idxx, config=self.config)
        loss2 = self.cut_loss(self.res(seg_pred2, self.base_size),
                              self.res(prop_pred1, self.base_size),
                              dic2, dic1, device, epoch=epoch, idxx=idxx, config=self.config)
        return loss1 + loss2

    def memory(self, prev_pred, prev_context4, prev_context3, prev_context2):
        pred = self.conv_m(prev_pred)
        k4, v4 = self.KV_M_r4(prev_context4 + self.ret(pred, prev_context4))
        k3, v3 = self.KV_M_r3(prev_context3 + self.ret(pred, prev_context3))
        k2, v2 = self.KV_M_r2(prev_context2 + self.ret(pred, prev_context2))

        return [k4.unsqueeze(2), v4.unsqueeze(2)], [k3.unsqueeze(2), v3.unsqueeze(2)], \
               [k2.unsqueeze(2), v2.unsqueeze(2)]

    def propagation(self, frame, Es_prop):
        """
        Args:
            frame: [b, t, c, h, w]
            Es_prop: [b, 1, h, w]

        Returns:
            list [b, 2, 1, h, w]
        """
        b, T, c, h, w = frame.shape
        frame_reshape = frame.reshape(b * T, c, h, w)

        # encoder
        with torch.no_grad():
            out, a, [x2_context, x3_context, x4_context], [x2, x3, x4] = \
                self.prop_encoder(frame_reshape)
            out.detach_()
            a.detach_()
            x2_context.detach_()
            x3_context.detach_()
            x4_context.detach_()
        out_reshape = out.reshape(b, T, *out.shape[1:])
        a_reshape = a.reshape(b, T, *a.shape[1:])
        x2_reshape = x2_context.reshape(b, T, *x2_context.shape[1:])
        x3_reshape = x3_context.reshape(b, T, *x3_context.shape[1:])
        x4_reshape = x4_context.reshape(b, T, *x4_context.shape[1:])

        allout2, allout3, allout4, allout5 = [], [], [], []
        allcrt2, allcrt3, allcrt4 = [], [], []

        prev_pred = Es_prop
        for t in range(1, T):
            # memorize
            prev_context4, prev_context3, prev_context2 = \
                x4_reshape[:, t - 1], x3_reshape[:, t - 1], x2_reshape[:, t - 1]
            context4, context3, context2 = x4_reshape[:, t], x3_reshape[:, t], x2_reshape[:, t]

            [prev_key4, prev_value4], [prev_key3, prev_value3], [prev_key2, prev_value2] = \
                self.memory(prev_pred, prev_context4, prev_context3, prev_context2)

            if t - 1 == 0:
                [this_keys4, this_values4], [this_keys3, this_values3], [this_keys2, this_values2] = \
                    [prev_key4, prev_value4], [prev_key3, prev_value3], [prev_key2, prev_value2]  # only prev memory
            else:
                this_keys4 = torch.cat([keys4, prev_key4], dim=2)
                this_values4 = torch.cat([values4, prev_value4], dim=2)
                this_keys3 = torch.cat([keys3, prev_key3], dim=2)
                this_values3 = torch.cat([values3, prev_value3], dim=2)
                this_keys2 = torch.cat([keys2, prev_key2], dim=2)
                this_values2 = torch.cat([values2, prev_value2], dim=2)

            # segment
            [out2, out3, out4, out5], [crt2, crt3, crt4] = self.prop_decoder(out_reshape[:, t], a_reshape[:, t],
                                                                             context4, context3, context2,
                                                                             this_keys4, this_values4, this_keys3,
                                                                             this_values3, this_keys2, this_values2,
                                                                             x4_reshape[:, 0], x3_reshape[:, 0],
                                                                             x2_reshape[:, 0], Es_prop)
            # update
            [keys4, values4], [keys3, values3], [keys2, values2] = \
                [this_keys4, this_values4], [this_keys3, this_values3], [this_keys2, this_values2]
            prev_pred = torch.sigmoid(out2.detach())

            allout2.append(out2)
            allout3.append(out3)
            allout4.append(out4)
            allout5.append(out5)
            allcrt2.append(crt2)
            allcrt3.append(crt3)
            allcrt4.append(crt4)

        return [torch.stack(allout2, dim=1), torch.stack(allout3, dim=1),
                torch.stack(allout4, dim=1), torch.stack(allout5, dim=1)], \
               [torch.stack(allcrt2, dim=1), torch.stack(allcrt3, dim=1),
                torch.stack(allcrt4, dim=1)]

    def segmentation(self, x):
        if len(x.shape) == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
        bt, c, h, w = x.shape
        b, T = bt // self.T, self.T
        base_shape = (1, 352, 352)

        out, a, [x2_context, x3_context, x4_context], [x2, x3, x4] = self.seg_encoder(x)
        [out2, out3, out4, out5], [corrtcted_map2, corrtcted_map3, corrtcted_map4] = \
            self.seg_decoder(out, a, x4_context, x3_context, x2_context)

        return [out2.reshape(b, T, *base_shape), out3.reshape(b, T, *base_shape),
                out4.reshape(b, T, *base_shape), out5.reshape(b, T, *base_shape)], \
               [corrtcted_map2.reshape(b, T, *base_shape), corrtcted_map3.reshape(b, T, *base_shape),
                corrtcted_map4.reshape(b, T, *base_shape)]

    def train_step_pretrain(self, Fs, Ms, Bs, device, epoch=0, idx=0):
        """
        Args:
            Fs: [b,t,3,h,w]
            Ms: [b,t,1,h,w]
        """
        b, T, c, h, w = Fs.shape
        Fs_reshape = Fs.reshape(b * T, c, h, w)
        Ms_reshape = Ms.reshape(b * T, 1, h, w)
        Bs_reshape = Bs.reshape(b * T, 1, h, w)

        # seg_branch
        seg_loss = 0
        pred_seg_list, crct_seg_list = self.segmentation(Fs)
        for t in range(T):
            for pred in pred_seg_list:
                seg_loss += self.ceiou_loss(pred[:, t], Ms[:, t])
            for map in crct_seg_list:
                seg_loss += self.ceiou_loss(map[:, t], Ms[:, t])
        seg_loss = seg_loss / T

        # prop_branch
        pred_prop_list, crct_prop_list = self.propagation(Fs, Ms[:, 0])
        # prop_loss
        prop_loss = 0
        for t in range(0, T - 1):
            for pred in pred_prop_list:
                prop_loss += self.ceiou_loss(pred[:, t], Ms[:, t + 1])
            for crt in crct_prop_list:
                prop_loss + self.ceiou_loss(crt[:, t], Ms[:, t + 1])
        prop_loss = prop_loss / (T - 1)

        if self.config.ifcut:
            cut_loss = self.context_free_pretrain(Fs_reshape, Ms_reshape, Bs_reshape, device, epoch=epoch, idxx=idx)

            loss_dic = {
                'seg': seg_loss,
                'prop': prop_loss,
                'byol': cut_loss,
                'total': seg_loss + prop_loss + cut_loss,
            }
        else:
            loss_dic = {
                'seg': seg_loss,
                'prop': prop_loss,
                'byol': prop_loss,
                'total': seg_loss + prop_loss,
            }

        pred_dic = {
            'seg': pred_seg_list,  # tensor bt, 1, h, w
            'prop': pred_prop_list,
            'crt_seg': crct_seg_list,
            'crt_prop': crct_prop_list,
        }
        return loss_dic, pred_dic

    def train_step_maintraining(self, Fs, Ms, Bs, device, epoch=0, idx=0):
        """
        Args:
            Fs: [b, t, c, h, w]
            Ms: [b, t, 1, h, w]
            Bs: [b, t, 1, h, w]
        Returns:
        """
        b, T, c, h, w = Fs.shape
        Fs_reshape = Fs.reshape(b * T, c, h, w)
        Ms_reshape = Ms.reshape(b * T, 1, h, w)
        Bs_reshape = Bs.reshape(b * T, 1, h, w)

        # seg_branch
        seg_loss = 0
        pred_seg_list, crct_seg_list = self.segmentation(Fs)
        for pred in pred_seg_list:
            seg_loss += self.ceiou_loss(pred[:, 0], Ms[:, 0])
        for map in crct_seg_list:
            seg_loss += self.ceiou_loss(map[:, 0], Ms[:, 0])

        # prop_branch
        pred_prop_list, crct_prop_list = self.propagation(Fs, Ms[:, 0])

        # cps_loss
        cps_loss = 0
        for t in range(0, T - 1):
            gt_prop = (torch.sigmoid(pred_prop_list[0][:, t].detach()) > 0.5).float()
            gt_seg = (torch.sigmoid(pred_seg_list[0][:, t + 1].detach()) > 0.5).float()
            # prop->seg
            for pred in pred_prop_list:
                cps_loss += self.ce_loss(pred[:, t], gt_seg)
            for crt in crct_prop_list:
                cps_loss + self.ce_loss(crt[:, t], gt_seg)
            # seg->prop
            for pred in pred_seg_list:
                cps_loss += self.ce_loss(pred[:, t + 1], gt_prop)
            for crct in crct_seg_list:
                cps_loss += self.ce_loss(crct[:, t + 1], gt_prop)

            if epoch < -1 and epoch % 1 == 0 and idx % 10 == 0:
                dic = {
                    'M_{}'.format(t + 1): Ms[0, t + 1, 0, ::],
                    'F_{}'.format(t + 1): Fs[0, t + 1, ::],
                    'B_{}'.format(t + 1): Bs[0, t + 1, ::],
                    'prop2_{}'.format(t + 1): torch.sigmoid(pred_prop_list[0][0, t, 0, ::]),
                    'seg2_{}'.format(t + 1): torch.sigmoid(pred_seg_list[0][0, t + 1, 0, ::]),
                    'propgt_{}'.format(t + 1): gt_prop[0, 0, ::],
                    'seggt_{}'.format(t + 1): gt_seg[0, 0, ::],
                }
                visualize(dic, self.config.visualize_path, epoch, idx)

        cps_loss = cps_loss / (T - 1)

        if self.config.ifcut:
            location, pseudo_label = get_location_dilate(torch.sigmoid(pred_seg_list[0]).detach(),
                                                         torch.sigmoid(pred_prop_list[0]).detach(), Ms[:, 0], device)
            cut_loss = self.context_free(Fs_reshape, location, Bs_reshape, pseudo_label, device, epoch=epoch, idxx=idx)

            loss_dic = {
                'seg': seg_loss,
                'cps': cps_loss,
                'byol': cut_loss,
                'total': seg_loss + self.config.cps_weight * cps_loss + cut_loss,
            }
        else:
            loss_dic = {
                'seg': seg_loss,
                'cps': cps_loss,
                'byol': cps_loss,
                'total': seg_loss + self.config.cps_weight * cps_loss,
            }
        pred_dic = {
            'seg': pred_seg_list,  # tensor 4* (bt, 1, h, w)
            'prop': pred_prop_list,  # list(list) t, b, 1, h, w
        }
        return loss_dic, pred_dic

    # def forward(self, Fs):
    #     preds, _ = self.segmentation(Fs)
    #     return preds[0]

    def my_eval(self, Fs):
        preds, _ = self.segmentation(Fs)
        return preds[0]

    # def my_eval_all(self, Fs, Ms, device):
    #     """
    #     Args:
    #         Fs: [b, t, c, h, w]
    #         Ms: [b, t, 1, h, w] 
    #         Bs: [b, t, 1, h, w]
    #     Returns:
    #     """
    #     b, T, c, h, w = Fs.shape
    #     Fs_reshape = Fs.reshape(b * T, c, h, w)
    #     Ms_reshape = Ms.reshape(b * T, 1, h, w)
    #     # seg_branch
    #     pred_seg_list, crct_seg_list = self.segmentation(Fs)
    #     pred_prop_list, crct_prop_list = self.propagation(Fs, Ms[:, 0])
    #
    #     pred_dic = {
    #         'seg': pred_seg_list,  # tensor bt, 1, h, w
    #         'prop': pred_prop_list,
    #         'crt_seg': crct_seg_list,
    #         'crt_prop': crct_prop_list,
    #     }
    #     return pred_dic

    # def test_cut(self, Fs, Ms, Bs, device, epoch=0, idx=0):
    #     """
    #     Args:
    #         Fs: [b, t, c, h, w]
    #         Ms: [b, t, 1, h, w] 
    #         Bs: [b, t, 1, h, w]
    #     Returns:
    #     """
    #     b, T, c, h, w = Fs.shape
    #     Fs_reshape = Fs.reshape(b * T, c, h, w)
    #     Ms_reshape = Ms.reshape(b * T, 1, h, w)
    #     Bs_reshape = Bs.reshape(b * T, 1, h, w)
    #     cut_loss = self.cut(Fs_reshape, Ms_reshape, Bs_reshape, Ms_reshape, device, epoch=epoch, idxx=idx)
    #     loss_dic = {
    #         'seg': cut_loss,
    #         'cps': cut_loss,
    #         'byol': cut_loss,
    #         'total': cut_loss,
    #     }
    #     pred_dic = {
    #         'seg': None,  # tensor 4* (bt, 1, h, w)
    #         'prop': None,  # list(list) t, b, 1, h, w
    #     }
    #     return loss_dic, pred_dic

    def save_checkpoint(self, path, name):
        state = {
            'step': self.total_step,
            'parameters': self.state_dict()
        }
        torch.save(state, '{}/{}.pth'.format(path, name))

    def load_checkpoint(self, path, logger=None):
        if logger is not None:
            logger.info("load model from {}".format(path))
        checkpoint = torch.load(path)
        self.total_step = checkpoint['step']

        trained_dict = checkpoint['parameters']
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in trained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = Backbone()
        self.context2 = RFB(512, 32)
        self.context3 = RFB(1024, 32)
        self.context4 = RFB(2048, 32)
        self.fusion = Fusion(32, outchannel=32)
        self.out_conv = conv(32, 1, 1, bn=False, bias=True)

    def forward(self, frame):
        x4, x3, x2, x1, _, _ = self.encoder(frame)  # 2048, 1024, 512,,r4 [b, 1024, h/16, w/16]
        x2_context = self.context2(x2)
        x3_context = self.context3(x3)
        x4_context = self.context4(x4)
        a = self.fusion(x4_context, x3_context, x2_context)
        out = self.out_conv(a)
        return out, a, [x2_context, x3_context, x4_context], [x2, x3, x4]


class Decoder_sqc(nn.Module):
    def __init__(self, channel, out_channel=1, k=8):
        super(Decoder_sqc, self).__init__()
        self.attention2 = SCRA(32, 32, channel=32, t=3, depth=2, kernel_size=3, k=k)
        self.attention3 = SCRA(32, 32, channel=32, t=3, depth=2, kernel_size=3, k=k)
        self.attention4 = SCRA(32, 32, channel=32, t=3, depth=2, kernel_size=3, k=k)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, out, fusion, x4, x3, x2):
        base_size = (352, 352)
        a5 = out
        out5 = self.res(a5, base_size)  # F.interpolate(a5, size=base_size, mode='bilinear', align_corners=False)

        f4, a4, corrtcted_map4 = self.attention4(x4, self.ret(fusion, x4), a5)
        out4 = self.res(a4, base_size)

        f3, a3, corrtcted_map3 = self.attention3(x3, self.ret(x4, x3), a4)
        out3 = self.res(a3, base_size)

        _, a2, corrtcted_map2 = self.attention2(x2, self.ret(x3, x2), a3)
        out2 = self.res(a2, base_size)

        corrtcted_map2 = self.res(corrtcted_map2, base_size)
        corrtcted_map3 = self.res(corrtcted_map3, base_size)
        corrtcted_map4 = self.res(corrtcted_map4, base_size)
        return [out2, out3, out4, out5], [corrtcted_map2, corrtcted_map3, corrtcted_map4]


class Decoder_prop_map(nn.Module):
    def __init__(self, channel, out_channel=1, k=8):
        super(Decoder_prop_map, self).__init__()

        self.attention_map2 = PCRA(32, m_channel=4, channel=32, t=3, depth=3, kernel_size=3, k=k)
        self.attention_map3 = PCRA(32, m_channel=4, channel=32, t=3, depth=3, kernel_size=3, k=k)
        self.attention_map4 = PCRA(32, m_channel=4, channel=32, t=3, depth=3, kernel_size=3, k=k)

        self.conv_pred4 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv_pred3 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv_pred2 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)

        self.KV_Q_r4 = KeyValue(32, keydim=4, valdim=16)
        self.KV_Q_r3 = KeyValue(32, keydim=4, valdim=16)
        self.KV_Q_r2 = KeyValue(32, keydim=4, valdim=16)
        self.Memory = Memory()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, out, a, context4, context3, context2,
                keys4, values4, keys3, values3, keys2, values2,
                x4_0, x3_0, x2_0, Es_0):
        """
        Args:
            context4:  context3: context2:
            keys4:  values4: keys3: values3: keys2: values2:
            x4_0:  x3_0: x2_0: Es_0p: [b, 1, h, w]
        Returns:

        """
        base_size = (352, 352)
        a5 = out
        out5 = self.res(a5, base_size)

        out5_m = self.conv_pred4(out5)
        k4, v4 = self.KV_Q_r4(context4 + self.ret(out5_m, context4))
        m4, viz4 = self.Memory(keys4, values4, k4, v4)
        _, a4, crt4 = self.attention_map4(m4, a5, context4, x4_0, Es_0)
        out4, crt4 = self.res(a4, base_size), self.res(crt4, base_size)

        out4_m = self.conv_pred3(out4)
        k3, v3 = self.KV_Q_r3(context3 + self.ret(out4_m, context3))
        m3, viz3 = self.Memory(keys3, values3, k3, v3)
        _, a3, crt3 = self.attention_map3(m3, a4, context3, x3_0, Es_0)
        out3, crt3 = self.res(a3, base_size), self.res(crt3, base_size)

        out2_m = self.conv_pred2(out3)
        k2, v2 = self.KV_Q_r2(context2 + self.ret(out2_m, context2))
        m2, viz2 = self.Memory(keys2, values2, k2, v2)
        _, a2, crt2 = self.attention_map2(m2, a3, context2, x2_0, Es_0)
        out2, crt2 = self.res(a2, base_size), self.res(crt2, base_size)

        return [out2, out3, out4, out5], [crt2, crt3, crt4]


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        # resnet = models.resnet50(pretrained=False)
        self.conv1 = feature_extractor.conv1
        self.bn1 = feature_extractor.bn1
        self.relu = feature_extractor.relu  # 1/2, 64
        self.maxpool = feature_extractor.maxpool

        self.res2 = feature_extractor.layer1  # 1/4, 256
        self.res3 = feature_extractor.layer2  # 1/8, 512
        self.res4 = feature_extractor.layer3  # 1/16, 1024
        self.res5 = feature_extractor.layer4  # 1/32, 2048

    def forward(self, in_f):
        x = self.conv1(in_f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024
        r5 = self.res5(r4)
        # 11, 22, 44,
        return r5, r4, r3, r2, c1, in_f


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        '''
        keys_M, values_M, k4_q, v4_q
        Args:
            m_in: key_M  [b, 128_pre, t, h/16, w/16]
            m_out: value_M, [b, 512, t_pre, h/16, w/16]
            q_in: key_q, [b, 128, h/16, h/16]
            q_out: value_q, [b, 512, h/16, w/16]

        Returns:
            mom_out [b, 1024, h, w]
            p [b, THW, HW]
        '''
        # print(m_in.shape, m_out.shape, q_out.shape, q_in.shape)
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T * H * W)  # b, 128 x THW
        mi = torch.transpose(mi, 1, 2)  # b, THW, 128 

        qi = q_in.view(B, D_e, H * W)  # b, 128, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_out.view(B, D_o, T * H * W)  # b, 512, THW
        mem = torch.bmm(mo, p)  # B, 512, HW
        mem = mem.view(B, D_o, H, W)  # b, 512, h, w

        mem_out = torch.cat([mem, q_out], dim=1)  # b, 1024, h, w

        return mem_out, p
