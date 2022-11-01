import torch
import torch.nn as nn
from torchvision.ops import nms, roi_align, roi_pool
import torch.nn.functional as F
import numpy as np
import random
from utils.tools import visualize
from model.losses import bce_iou_loss


class syn_loss(nn.Module):
    def __init__(self):
        super(syn_loss, self).__init__()
        self.crop = 192
        self.crr_crop = 5  

    def loss_fn(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss

    def ce_loss(self, x, y):
        ce = nn.BCEWithLogitsLoss()
        return ce(x.unsqueeze(0), y.unsqueeze(0))

    def ceiou_loss(self, x, y):
        ceiou = bce_iou_loss
        return ceiou(x.unsqueeze(0), y.unsqueeze(0))

    def forward(self, x, y, dic1, dic2, device, epoch=0, idxx=0, config=None):
        syn1, syn2 = dic1['view'], dic2['view']
        crop_box1, crop_box2 = dic1['crop_box'], dic2['crop_box']
        over_box1, over_box2 = dic1['overlap_box'], dic2['overlap_box']
        sbor1, sbor2 = dic1['sbor'], dic2['sbor'] 
        label1, label2 = dic1['label'], dic2['label']
        b, c, h, w = x.shape
        loss = 0
        for idx in range(x.size(0)):
            # print('box', crop_box1[idx], crop_box2[idx], over_box1[idx], over_box2[idx])
            [_, crop1_x1, crop1_y1, crop1_x2, crop1_y2] = crop_box1[idx]
            [_, crop2_x1, crop2_y1, crop2_x2, crop2_y2] = crop_box2[idx]
            [_, over1_x1, over1_y1, over1_x2, over1_y2] = over_box1[idx]
            [_, over2_x1, over2_y1, over2_x2, over2_y2] = over_box2[idx]

            crop1_img = syn1[idx, :, crop1_y1:crop1_y2, crop1_x1:crop1_x2]
            crop2_img = syn2[idx, :, crop2_y1:crop2_y2, crop2_x1:crop2_x2]
            over1_img = crop1_img[:, over1_y1:over1_y2, over1_x1:over1_x2]
            over2_img = crop2_img[:, over2_y1:over2_y2, over2_x1:over2_x2]

            crop1 = x[idx, :, crop1_y1:crop1_y2, crop1_x1:crop1_x2]
            crop2 = y[idx, :, crop2_y1:crop2_y2, crop2_x1:crop2_x2]

            over1_idx = crop1[:, over1_y1:over1_y2, over1_x1:over1_x2]
            over2_idx = crop2[:, over2_y1:over2_y2, over2_x1:over2_x2]

            bcrop1 = sbor1[idx, :, crop1_y1:crop1_y2, crop1_x1:crop1_x2].to(device)
            bcrop2 = sbor2[idx, :, crop2_y1:crop2_y2, crop2_x1:crop2_x2].to(device)
            bover1_idx = bcrop1[:, over1_y1:over1_y2, over1_x1:over1_x2]
            bover2_idx = bcrop2[:, over2_y1:over2_y2, over2_x1:over2_x2]
            correctional_over1_idx = over1_idx * bover1_idx
            correctional_over2_idx = over2_idx * bover2_idx
            correctional_crop1_idx = crop1 * bcrop1
            correctional_label1_idx = label1[idx] * bcrop1

            v1_idx = correctional_over1_idx.view(-1)
            v2_idx = correctional_over2_idx.view(-1)

            loss_idxx = self.loss_fn(torch.sigmoid(v1_idx), torch.sigmoid(v2_idx))

            if np.isnan(loss_idxx.item()):
                print("\n\nerror\n\n", over_box1[idx], over_box2[idx], '\n\n')
                loss_idxx = 0

            loss += loss_idxx

            if epoch < -1 and epoch % 5 == 0 and idxx == 0 and idx == 0:
                dic = {
                    'loss-crr-crop': torch.sigmoid(correctional_crop1_idx),
                    'loss-crr-label': correctional_label1_idx,

                    'loss-synimg1': syn1[idx],
                    'loss-crop-img1': crop1_img,
                    'loss-over-img1': over1_img,
                    'loss-x': torch.sigmoid(x[idx]),
                    'loss-crop1': torch.sigmoid(crop1),
                    'loss-over1': torch.sigmoid(over1_idx),
                    'loss-bc1': bcrop1,
                    'loss-bo1': bover1_idx,
                    'loss-crr1': torch.sigmoid(correctional_over1_idx),
                    'loss-label1': label1[idx],
                }
                visualize(dic, config.visualize_path + '/cut', epoch, str(idxx) + '-' + str(idx))

        return loss / b

